import gradio as gr
import numpy as np
import random
import torch
import spaces
import base64
from io import BytesIO

from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
#from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

import os
from gradio_client import Client, handle_file
import tempfile
from typing import Optional, Tuple, Any


# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder='transformer',
        torch_dtype=dtype,
        device_map='cuda'
    ),
    torch_dtype=dtype
).to(device)

pipe.load_lora_weights(
    "dx8152/Qwen-Edit-2509-Multiple-angles",
    weight_name="镜头转换.safetensors",
    adapter_name="angles"
)

pipe.set_adapters(["angles"], adapter_weights=[1.])
pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
pipe.unload_lora_weights()

spaces.aoti_blocks_load(pipe.transformer, "zerogpu-aoti/Qwen-Image", variant="fa3")

MAX_SEED = np.iinfo(np.int32).max


def _generate_video_segment(
    input_image_path: str,
    output_image_path: str,
    prompt: str,
    request: gr.Request
) -> str:
    """Generate a single video segment between two frames."""
    x_ip_token = request.headers['x-ip-token']
    video_client = Client(
        "multimodalart/wan-2-2-first-last-frame",
        headers={"x-ip-token": x_ip_token}
    )
    result = video_client.predict(
        start_image_pil=handle_file(input_image_path),
        end_image_pil=handle_file(output_image_path),
        prompt=prompt,
        api_name="/generate_video",
    )
    return result[0]["video"]


def build_camera_prompt(
    rotate_deg: float = 0.0,
    move_forward: float = 0.0,
    vertical_tilt: float = 0.0,
    wideangle: bool = False
) -> str:
    """Build a camera movement prompt based on the chosen controls."""
    prompt_parts = []

    if rotate_deg != 0:
        direction = "left" if rotate_deg > 0 else "right"
        if direction == "left":
            prompt_parts.append(
                f"将镜头向左旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the left."
            )
        else:
            prompt_parts.append(
                f"将镜头向右旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the right."
            )

    if move_forward > 5:
        prompt_parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
    elif move_forward >= 1:
        prompt_parts.append("将镜头向前移动 Move the camera forward.")

    if vertical_tilt <= -1:
        prompt_parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
    elif vertical_tilt >= 1:
        prompt_parts.append("将相机切换到仰视视角 Turn the camera to a worm's-eye view.")

    if wideangle:
        prompt_parts.append("将镜头转为广角镜头 Turn the camera to a wide-angle lens.")

    final_prompt = " ".join(prompt_parts).strip()
    return final_prompt if final_prompt else "no camera movement"


@spaces.GPU
def infer_camera_edit(
    image: Optional[Image.Image] = None,
    rotate_deg: float = 0.0,
    move_forward: float = 0.0,
    vertical_tilt: float = 0.0,
    wideangle: bool = False,
    seed: int = 0,
    randomize_seed: bool = True,
    true_guidance_scale: float = 1.0,
    num_inference_steps: int = 4,
    height: Optional[int] = None,
    width: Optional[int] = None,
    prev_output: Optional[Image.Image] = None,
) -> Tuple[Image.Image, int, str]:
    """Edit the camera angles/view of an image with Qwen Image Edit 2509."""
    progress = gr.Progress(track_tqdm=True)
    
    prompt = build_camera_prompt(rotate_deg, move_forward, vertical_tilt, wideangle)
    print(f"Generated Prompt: {prompt}")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    pil_images = []
    if image is not None:
        if isinstance(image, Image.Image):
            pil_images.append(image.convert("RGB"))
        elif hasattr(image, "name"):
            pil_images.append(Image.open(image.name).convert("RGB"))
    elif prev_output:
        pil_images.append(prev_output.convert("RGB"))

    if len(pil_images) == 0:
        raise gr.Error("Please upload an image first.")

    if prompt == "no camera movement":
        return image, seed, prompt

    result = pipe(
        image=pil_images,
        prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    return result, seed, prompt


def create_video_between_images(
    input_image: Optional[Image.Image],
    output_image: Optional[np.ndarray],
    prompt: str,
    request: gr.Request
) -> str:
    """Create a short transition video between the input and output images."""
    if input_image is None or output_image is None:
        raise gr.Error("Both input and output images are required to create a video.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            input_image.save(tmp.name)
            input_image_path = tmp.name

        output_pil = Image.fromarray(output_image.astype('uint8'))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            output_pil.save(tmp.name)
            output_image_path = tmp.name

        video_path = _generate_video_segment(
            input_image_path,
            output_image_path,
            prompt if prompt else "Camera movement transformation",
            request
        )
        return video_path
    except Exception as e:
        raise gr.Error(f"Video generation failed: {e}")


# --- 3D Camera Control Component for 2509 ---
CAMERA_3D_HTML_TEMPLATE = """
<div id="camera-control-wrapper" style="width: 100%; height: 400px; position: relative; background: #1a1a1a; border-radius: 12px; overflow: hidden;">
    <div id="prompt-overlay" style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); padding: 8px 16px; border-radius: 8px; font-family: monospace; font-size: 11px; color: #00ff88; white-space: nowrap; z-index: 10; max-width: 90%; overflow: hidden; text-overflow: ellipsis;"></div>
    <div id="control-legend" style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 8px 12px; border-radius: 8px; font-family: system-ui; font-size: 11px; color: #fff; z-index: 10;">
        <div style="margin-bottom: 4px;"><span style="color: #00ff88;">●</span> Rotation (↔)</div>
        <div style="margin-bottom: 4px;"><span style="color: #ff69b4;">●</span> Vertical Tilt (↕)</div>
        <div><span style="color: #ffa500;">●</span> Distance/Zoom</div>
    </div>
</div>
"""

CAMERA_3D_JS = """
(() => {
    const wrapper = element.querySelector('#camera-control-wrapper');
    const promptOverlay = element.querySelector('#prompt-overlay');
    
    const initScene = () => {
        if (typeof THREE === 'undefined') {
            setTimeout(initScene, 100);
            return;
        }
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);
        
        const camera = new THREE.PerspectiveCamera(50, wrapper.clientWidth / wrapper.clientHeight, 0.1, 1000);
        camera.position.set(4, 3, 4);
        camera.lookAt(0, 0.75, 0);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        wrapper.insertBefore(renderer.domElement, wrapper.firstChild);
        
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight.position.set(5, 10, 5);
        scene.add(dirLight);
        
        scene.add(new THREE.GridHelper(6, 12, 0x333333, 0x222222));
        
        const CENTER = new THREE.Vector3(0, 0.75, 0);
        const BASE_DISTANCE = 2.0;
        const ROTATION_RADIUS = 2.2;
        const TILT_RADIUS = 1.6;
        
        let rotateDeg = props.value?.rotate_deg || 0;
        let moveForward = props.value?.move_forward || 0;
        let verticalTilt = props.value?.vertical_tilt || 0;
        let wideangle = props.value?.wideangle || false;
        
        const rotateSteps = [-90, -45, 0, 45, 90];
        const forwardSteps = [0, 5, 10];
        const tiltSteps = [-1, 0, 1];
        
        function snapToNearest(value, steps) {
            return steps.reduce((prev, curr) => Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev);
        }
        
        function createPlaceholderTexture() {
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 256;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#3a3a4a';
            ctx.fillRect(0, 0, 256, 256);
            ctx.fillStyle = '#ffcc99';
            ctx.beginPath();
            ctx.arc(128, 128, 80, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#333';
            ctx.beginPath();
            ctx.arc(100, 110, 10, 0, Math.PI * 2);
            ctx.arc(156, 110, 10, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(128, 130, 35, 0.2, Math.PI - 0.2);
            ctx.stroke();
            return new THREE.CanvasTexture(canvas);
        }
        
        let currentTexture = createPlaceholderTexture();
        const planeMaterial = new THREE.MeshBasicMaterial({ map: currentTexture, side: THREE.DoubleSide });
        let targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
        targetPlane.position.copy(CENTER);
        scene.add(targetPlane);
        
        function updateTextureFromUrl(url) {
            if (!url) {
                planeMaterial.map = createPlaceholderTexture();
                planeMaterial.needsUpdate = true;
                scene.remove(targetPlane);
                targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                targetPlane.position.copy(CENTER);
                scene.add(targetPlane);
                return;
            }
            
            const loader = new THREE.TextureLoader();
            loader.crossOrigin = 'anonymous';
            loader.load(url, (texture) => {
                texture.minFilter = THREE.LinearFilter;
                texture.magFilter = THREE.LinearFilter;
                planeMaterial.map = texture;
                planeMaterial.needsUpdate = true;
                
                const img = texture.image;
                if (img && img.width && img.height) {
                    const aspect = img.width / img.height;
                    const maxSize = 1.4;
                    let planeWidth, planeHeight;
                    if (aspect > 1) {
                        planeWidth = maxSize;
                        planeHeight = maxSize / aspect;
                    } else {
                        planeHeight = maxSize;
                        planeWidth = maxSize * aspect;
                    }
                    scene.remove(targetPlane);
                    targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(planeWidth, planeHeight), planeMaterial);
                    targetPlane.position.copy(CENTER);
                    scene.add(targetPlane);
                }
            });
        }
        
        if (props.imageUrl) {
            updateTextureFromUrl(props.imageUrl);
        }
        
        const cameraGroup = new THREE.Group();
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 });
        const body = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.2, 0.35), bodyMat);
        cameraGroup.add(body);
        const lens = new THREE.Mesh(
            new THREE.CylinderGeometry(0.08, 0.1, 0.16, 16),
            new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 })
        );
        lens.rotation.x = Math.PI / 2;
        lens.position.z = 0.24;
        cameraGroup.add(lens);
        scene.add(cameraGroup);
        
        const rotationArcPoints = [];
        for (let i = 0; i <= 32; i++) {
            const angle = THREE.MathUtils.degToRad(-90 + (180 * i / 32));
            rotationArcPoints.push(new THREE.Vector3(ROTATION_RADIUS * Math.sin(angle), 0.05, ROTATION_RADIUS * Math.cos(angle)));
        }
        const rotationCurve = new THREE.CatmullRomCurve3(rotationArcPoints);
        const rotationArc = new THREE.Mesh(
            new THREE.TubeGeometry(rotationCurve, 32, 0.035, 8, false),
            new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.3 })
        );
        scene.add(rotationArc);
        
        const rotationHandle = new THREE.Mesh(
            new THREE.SphereGeometry(0.16, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.5 })
        );
        rotationHandle.userData.type = 'rotation';
        scene.add(rotationHandle);
        
        const tiltArcPoints = [];
        for (let i = 0; i <= 32; i++) {
            const angle = THREE.MathUtils.degToRad(-45 + (90 * i / 32));
            tiltArcPoints.push(new THREE.Vector3(-0.7, TILT_RADIUS * Math.sin(angle) + CENTER.y, TILT_RADIUS * Math.cos(angle)));
        }
        const tiltCurve = new THREE.CatmullRomCurve3(tiltArcPoints);
        const tiltArc = new THREE.Mesh(
            new THREE.TubeGeometry(tiltCurve, 32, 0.035, 8, false),
            new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.3 })
        );
        scene.add(tiltArc);
        
        const tiltHandle = new THREE.Mesh(
            new THREE.SphereGeometry(0.16, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.5 })
        );
        tiltHandle.userData.type = 'tilt';
        scene.add(tiltHandle);
        
        const distanceLineGeo = new THREE.BufferGeometry();
        const distanceLine = new THREE.Line(distanceLineGeo, new THREE.LineBasicMaterial({ color: 0xffa500 }));
        scene.add(distanceLine);
        
        const distanceHandle = new THREE.Mesh(
            new THREE.SphereGeometry(0.16, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0xffa500, emissive: 0xffa500, emissiveIntensity: 0.5 })
        );
        distanceHandle.userData.type = 'distance';
        scene.add(distanceHandle);
        
        function buildPromptText(rot, fwd, tilt, wide) {
            const parts = [];
            if (rot !== 0) {
                const dir = rot > 0 ? 'left' : 'right';
                parts.push('Rotate ' + Math.abs(rot) + '° ' + dir);
            }
            if (fwd > 5) parts.push('Close-up');
            else if (fwd >= 1) parts.push('Move forward');
            if (tilt <= -1) parts.push("Bird's-eye");
            else if (tilt >= 1) parts.push("Worm's-eye");
            if (wide) parts.push('Wide-angle');
            return parts.length > 0 ? parts.join(' • ') : 'No camera movement';
        }
        
        function updatePositions() {
            const rotRad = THREE.MathUtils.degToRad(-rotateDeg);
            const distance = BASE_DISTANCE - (moveForward / 10) * 1.0;
            // Invert: worm's-eye (1) = camera DOWN, bird's-eye (-1) = camera UP
            const tiltAngle = -verticalTilt * 35;
            const tiltRad = THREE.MathUtils.degToRad(tiltAngle);
            
            const camX = distance * Math.sin(rotRad) * Math.cos(tiltRad);
            const camY = distance * Math.sin(tiltRad) + CENTER.y;
            const camZ = distance * Math.cos(rotRad) * Math.cos(tiltRad);
            
            cameraGroup.position.set(camX, camY, camZ);
            cameraGroup.lookAt(CENTER);
            
            rotationHandle.position.set(ROTATION_RADIUS * Math.sin(rotRad), 0.05, ROTATION_RADIUS * Math.cos(rotRad));
            
            const tiltHandleAngle = THREE.MathUtils.degToRad(tiltAngle);
            tiltHandle.position.set(-0.7, TILT_RADIUS * Math.sin(tiltHandleAngle) + CENTER.y, TILT_RADIUS * Math.cos(tiltHandleAngle));
            
            const handleDist = distance - 0.4;
            distanceHandle.position.set(
                handleDist * Math.sin(rotRad) * Math.cos(tiltRad),
                handleDist * Math.sin(tiltRad) + CENTER.y,
                handleDist * Math.cos(rotRad) * Math.cos(tiltRad)
            );
            distanceLineGeo.setFromPoints([cameraGroup.position.clone(), CENTER.clone()]);
            
            promptOverlay.textContent = buildPromptText(rotateDeg, moveForward, verticalTilt, wideangle);
        }
        
        function updatePropsAndTrigger() {
            const rotSnap = snapToNearest(rotateDeg, rotateSteps);
            const fwdSnap = snapToNearest(moveForward, forwardSteps);
            const tiltSnap = snapToNearest(verticalTilt, tiltSteps);
            
            props.value = { rotate_deg: rotSnap, move_forward: fwdSnap, vertical_tilt: tiltSnap, wideangle: wideangle };
            trigger('change', props.value);
        }
        
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        let isDragging = false;
        let dragTarget = null;
        let dragStartMouse = new THREE.Vector2();
        let dragStartForward = 0;
        const intersection = new THREE.Vector3();
        
        const canvas = renderer.domElement;
        
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects([rotationHandle, tiltHandle, distanceHandle]);
            
            if (intersects.length > 0) {
                isDragging = true;
                dragTarget = intersects[0].object;
                dragTarget.material.emissiveIntensity = 1.0;
                dragTarget.scale.setScalar(1.3);
                dragStartMouse.copy(mouse);
                dragStartForward = moveForward;
                canvas.style.cursor = 'grabbing';
            }
        });
        
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            
            if (isDragging && dragTarget) {
                raycaster.setFromCamera(mouse, camera);
                
                if (dragTarget.userData.type === 'rotation') {
                    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        let angle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                        rotateDeg = THREE.MathUtils.clamp(-angle, -90, 90);
                    }
                } else if (dragTarget.userData.type === 'tilt') {
                    const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0.7);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        const relY = intersection.y - CENTER.y;
                        const relZ = intersection.z;
                        const angle = THREE.MathUtils.radToDeg(Math.atan2(relY, relZ));
                        // Invert: drag DOWN = worm's-eye (1), drag UP = bird's-eye (-1)
                        verticalTilt = THREE.MathUtils.clamp(-angle / 35, -1, 1);
                    }
                } else if (dragTarget.userData.type === 'distance') {
                    const deltaY = mouse.y - dragStartMouse.y;
                    moveForward = THREE.MathUtils.clamp(dragStartForward + deltaY * 12, 0, 10);
                }
                updatePositions();
            } else {
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects([rotationHandle, tiltHandle, distanceHandle]);
                [rotationHandle, tiltHandle, distanceHandle].forEach(h => {
                    h.material.emissiveIntensity = 0.5;
                    h.scale.setScalar(1);
                });
                if (intersects.length > 0) {
                    intersects[0].object.material.emissiveIntensity = 0.8;
                    intersects[0].object.scale.setScalar(1.1);
                    canvas.style.cursor = 'grab';
                } else {
                    canvas.style.cursor = 'default';
                }
            }
        });
        
        const onMouseUp = () => {
            if (dragTarget) {
                dragTarget.material.emissiveIntensity = 0.5;
                dragTarget.scale.setScalar(1);
                
                const targetRot = snapToNearest(rotateDeg, rotateSteps);
                const targetFwd = snapToNearest(moveForward, forwardSteps);
                const targetTilt = snapToNearest(verticalTilt, tiltSteps);
                
                const startRot = rotateDeg, startFwd = moveForward, startTilt = verticalTilt;
                const startTime = Date.now();
                
                function animateSnap() {
                    const t = Math.min((Date.now() - startTime) / 200, 1);
                    const ease = 1 - Math.pow(1 - t, 3);
                    
                    rotateDeg = startRot + (targetRot - startRot) * ease;
                    moveForward = startFwd + (targetFwd - startFwd) * ease;
                    verticalTilt = startTilt + (targetTilt - startTilt) * ease;
                    
                    updatePositions();
                    if (t < 1) requestAnimationFrame(animateSnap);
                    else updatePropsAndTrigger();
                }
                animateSnap();
            }
            isDragging = false;
            dragTarget = null;
            canvas.style.cursor = 'default';
        };
        
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('mouseleave', onMouseUp);

        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects([rotationHandle, tiltHandle, distanceHandle]);
            
            if (intersects.length > 0) {
                isDragging = true;
                dragTarget = intersects[0].object;
                dragTarget.material.emissiveIntensity = 1.0;
                dragTarget.scale.setScalar(1.3);
                dragStartMouse.copy(mouse);
                dragStartForward = moveForward;
            }
        }, { passive: false });
        
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
            
            if (isDragging && dragTarget) {
                raycaster.setFromCamera(mouse, camera);
                
                if (dragTarget.userData.type === 'rotation') {
                    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        let angle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                        rotateDeg = THREE.MathUtils.clamp(-angle, -90, 90);
                    }
                } else if (dragTarget.userData.type === 'tilt') {
                    const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0.7);
                    if (raycaster.ray.intersectPlane(plane, intersection)) {
                        const relY = intersection.y - CENTER.y;
                        const relZ = intersection.z;
                        const angle = THREE.MathUtils.radToDeg(Math.atan2(relY, relZ));
                        // Invert: drag DOWN = worm's-eye (1), drag UP = bird's-eye (-1)
                        verticalTilt = THREE.MathUtils.clamp(-angle / 35, -1, 1);
                    }
                } else if (dragTarget.userData.type === 'distance') {
                    const deltaY = mouse.y - dragStartMouse.y;
                    moveForward = THREE.MathUtils.clamp(dragStartForward + deltaY * 12, 0, 10);
                }
                updatePositions();
            }
        }, { passive: false });
        
        canvas.addEventListener('touchend', (e) => { e.preventDefault(); onMouseUp(); }, { passive: false });
        canvas.addEventListener('touchcancel', (e) => { e.preventDefault(); onMouseUp(); }, { passive: false });
        
        updatePositions();
        
        function render() {
            requestAnimationFrame(render);
            renderer.render(scene, camera);
        }
        render();
        
        new ResizeObserver(() => {
            camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
        }).observe(wrapper);
        
        wrapper._updateTexture = updateTextureFromUrl;
        
        let lastImageUrl = props.imageUrl;
        let lastValue = JSON.stringify(props.value);
        setInterval(() => {
            if (props.imageUrl !== lastImageUrl) {
                lastImageUrl = props.imageUrl;
                updateTextureFromUrl(props.imageUrl);
            }
            const currentValue = JSON.stringify(props.value);
            if (currentValue !== lastValue) {
                lastValue = currentValue;
                if (props.value && typeof props.value === 'object') {
                    rotateDeg = props.value.rotate_deg ?? rotateDeg;
                    moveForward = props.value.move_forward ?? moveForward;
                    verticalTilt = props.value.vertical_tilt ?? verticalTilt;
                    wideangle = props.value.wideangle ?? wideangle;
                    updatePositions();
                }
            }
        }, 100);
    };
    
    initScene();
})();
"""


def create_camera_3d_component(value=None, imageUrl=None, **kwargs):
    """Create a 3D camera control component using gr.HTML."""
    if value is None:
        value = {"rotate_deg": 0, "move_forward": 0, "vertical_tilt": 0, "wideangle": False}
    
    return gr.HTML(
        value=value,
        html_template=CAMERA_3D_HTML_TEMPLATE,
        js_on_load=CAMERA_3D_JS,
        imageUrl=imageUrl,
        **kwargs
    )


# --- UI ---
css = '''
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .progress-text { color: white !important; }
#camera-3d-control { min-height: 400px; }
#examples {
    margin-top: 20px;
}
.fillable{max-width: 1200px !important}
'''


def reset_all() -> list:
    """Reset all camera control knobs and flags to their default values."""
    return [0, 0, 0, False, True]


def end_reset() -> bool:
    """Mark the end of a reset cycle."""
    return False


def update_dimensions_on_upload(image: Optional[Image.Image]) -> Tuple[int, int]:
    """Compute recommended (width, height) for the output resolution."""
    if image is None:
        return 1024, 1024

    original_width, original_height = image.size

    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)

    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    return new_width, new_height


with gr.Blocks() as demo:
    gr.Markdown("""
    ## 🎬 Qwen Image Edit — Camera Angle Control
    
    Qwen Image Edit 2509 for Camera Control ✨ 
    Using [dx8152's Qwen-Edit-2509-Multiple-angles LoRA](https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles) and [Phr00t/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main) for 4-step inference 💨
    """)

    with gr.Row():
        with gr.Column(scale=1):
            
            image = gr.Image(label="Input Image", type="pil", height=280)
            prev_output = gr.Image(value=None, visible=False)
            is_reset = gr.Checkbox(value=False, visible=False)
            
            with gr.Tab("🎮 3D Camera Control"):
                # gr.Markdown("*Drag the handles: 🟢 Rotation, 🩷 Tilt, 🟠 Distance*")
                
                camera_3d = create_camera_3d_component(
                    value={"rotate_deg": 0, "move_forward": 0, "vertical_tilt": 0, "wideangle": False},
                    elem_id="camera-3d-control"
                )
            with gr.Tab("🎚️ Slider Controls"):
                rotate_deg = gr.Slider(label="Rotate Right ↔ Left (°)", minimum=-90, maximum=90, step=45, value=0)
                move_forward = gr.Slider(label="Move Forward → Close-Up", minimum=0, maximum=10, step=5, value=0)
                vertical_tilt = gr.Slider(label="Vertical: Bird's-eye ↔ Worm's-eye", minimum=-1, maximum=1, step=1, value=0)
                wideangle = gr.Checkbox(label="🔭 Wide-Angle Lens", value=False)
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset")
                run_btn = gr.Button("🚀 Generate", variant="primary")
        
        with gr.Column(scale=1):
            result = gr.Image(label="Output Image", interactive=False, height=350)
            prompt_preview = gr.Textbox(label="Generated Prompt", interactive=False)
            
            create_video_button = gr.Button(
                "🎥 Create Video Between Images",
                variant="secondary",
                visible=False
            )
            with gr.Group(visible=False) as video_group:
                video_output = gr.Video(label="Generated Video", buttons=["download"], autoplay=True)
        
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                true_guidance_scale = gr.Slider(label="True Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=40, step=1, value=4)
                height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=1024)
                width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=1024)

    # --- Helper Functions ---
    def update_prompt_from_sliders(rotate, forward, tilt, wide):
        return build_camera_prompt(rotate, forward, tilt, wide)
    
    def sync_3d_to_sliders(camera_value):
        if camera_value and isinstance(camera_value, dict):
            rot = camera_value.get('rotate_deg', 0)
            fwd = camera_value.get('move_forward', 0)
            tilt = camera_value.get('vertical_tilt', 0)
            wide = camera_value.get('wideangle', False)
            prompt = build_camera_prompt(rot, fwd, tilt, wide)
            return rot, fwd, tilt, wide, prompt
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    def sync_sliders_to_3d(rotate, forward, tilt, wide):
        return {"rotate_deg": rotate, "move_forward": forward, "vertical_tilt": tilt, "wideangle": wide}
    
    def update_3d_image(img):
        if img is None:
            return gr.update(imageUrl=None)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        return gr.update(imageUrl=data_url)
    
    # Define inputs/outputs
    inputs = [image, rotate_deg, move_forward, vertical_tilt, wideangle, seed, randomize_seed, true_guidance_scale, num_inference_steps, height, width, prev_output]
    outputs = [result, seed, prompt_preview]
    control_inputs = [image, rotate_deg, move_forward, vertical_tilt, wideangle, seed, randomize_seed, true_guidance_scale, num_inference_steps, height, width, prev_output]
    control_inputs_with_flag = [is_reset] + control_inputs
    
    def maybe_infer(is_reset_val: bool, progress: gr.Progress = gr.Progress(track_tqdm=True), *args: Any):
        if is_reset_val:
            return gr.update(), gr.update(), gr.update(), gr.update()
        result_img, result_seed, result_prompt = infer_camera_edit(*args)
        show_button = args[0] is not None and result_img is not None
        return result_img, result_seed, result_prompt, gr.update(visible=show_button)
    
    # --- Event Handlers ---
    
    # Slider -> Prompt preview
    for slider in [rotate_deg, move_forward, vertical_tilt]:
        slider.change(fn=update_prompt_from_sliders, inputs=[rotate_deg, move_forward, vertical_tilt, wideangle], outputs=[prompt_preview])
    wideangle.change(fn=update_prompt_from_sliders, inputs=[rotate_deg, move_forward, vertical_tilt, wideangle], outputs=[prompt_preview])
    
    # 3D control -> Sliders + Prompt + Inference
    camera_3d.change(
        fn=sync_3d_to_sliders,
        inputs=[camera_3d],
        outputs=[rotate_deg, move_forward, vertical_tilt, wideangle, prompt_preview]
    ).then(
        fn=maybe_infer,
        inputs=control_inputs_with_flag,
        outputs=outputs + [create_video_button]
    )
    
    # Sliders -> 3D control
    for slider in [rotate_deg, move_forward, vertical_tilt]:
        slider.release(fn=sync_sliders_to_3d, inputs=[rotate_deg, move_forward, vertical_tilt, wideangle], outputs=[camera_3d])
    wideangle.input(fn=sync_sliders_to_3d, inputs=[rotate_deg, move_forward, vertical_tilt, wideangle], outputs=[camera_3d])
    
    # Reset
    reset_btn.click(fn=reset_all, inputs=None, outputs=[rotate_deg, move_forward, vertical_tilt, wideangle, is_reset], queue=False
    ).then(fn=end_reset, inputs=None, outputs=[is_reset], queue=False
    ).then(fn=sync_sliders_to_3d, inputs=[rotate_deg, move_forward, vertical_tilt, wideangle], outputs=[camera_3d])
    
    # Generate button
    def infer_and_show_video_button(*args: Any):
        result_img, result_seed, result_prompt = infer_camera_edit(*args)
        show_button = args[0] is not None and result_img is not None
        return result_img, result_seed, result_prompt, gr.update(visible=show_button)
    
    run_event = run_btn.click(fn=infer_and_show_video_button, inputs=inputs, outputs=outputs + [create_video_button])
    
    # Video creation
    create_video_button.click(fn=lambda: gr.update(visible=True), outputs=[video_group], api_visibility="private"
    ).then(fn=create_video_between_images, inputs=[image, result, prompt_preview], outputs=[video_output], api_visibility="private")
    
    # Image upload
    image.upload(fn=update_dimensions_on_upload, inputs=[image], outputs=[width, height]
    ).then(fn=reset_all, inputs=None, outputs=[rotate_deg, move_forward, vertical_tilt, wideangle, is_reset], queue=False
    ).then(fn=end_reset, inputs=None, outputs=[is_reset], queue=False
    ).then(fn=update_3d_image, inputs=[image], outputs=[camera_3d])
    
    image.clear(fn=lambda: gr.update(imageUrl=None), outputs=[camera_3d])

    run_event.then(lambda img, *_: img, inputs=[result], outputs=[prev_output])


    gr.Examples(
        examples=[
            ["tool_of_the_sea.png", 90, 0, 0, False, 0, True, 1.0, 4, 568, 1024],
            ["monkey.jpg", -90, 0, 0, False, 0, True, 1.0, 4, 704, 1024],
            ["metropolis.jpg", 0, 0, -1, False, 0, True, 1.0, 4, 816, 1024],
            ["disaster_girl.jpg", -45, 0, 1, False, 0, True, 1.0, 4, 768, 1024],
            ["grumpy.png", 90, 0, 1, False, 0, True, 1.0, 4, 576, 1024]
        ],
        inputs=[image, rotate_deg, move_forward, vertical_tilt, wideangle, seed, randomize_seed, true_guidance_scale, num_inference_steps, height, width],
        outputs=outputs,
        fn=infer_camera_edit,
        cache_examples=True,
        cache_mode="lazy",
        elem_id="examples"
    )
    
    gr.api(infer_camera_edit, api_name="infer_edit_camera_angles")
    gr.api(create_video_between_images, api_name="create_video_between_images")

head = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
demo.launch(mcp_server=True, css=css, theme=gr.themes.Citrus(), head=head, footer_links=["api", "gradio", "settings"])