/**
 * 3D Hand Visualization - Iron Man Energy Gauntlet
 * =================================================
 * Creates a futuristic Iron Man-style armored gauntlet around the hand:
 * - Metallic armor plates on each finger segment
 * - Glowing repulsor in the palm
 * - Arc reactor energy lines between joints
 * - Pulsing power indicators
 * - Holographic HUD elements
 */

// ============== CONFIGURATION ==============
const CONFIG = {
    websocketUrl: 'ws://localhost:8765',
    reconnectInterval: 2000,

    // Iron Man color schemes
    themes: {
        mark3: {
            primary: 0xb91c1c,      // Red armor
            secondary: 0xfbbf24,    // Gold trim
            energy: 0x38bdf8,       // Arc reactor blue
            glow: 0x0ea5e9,         // Energy glow
            metal: 0x991b1b
        },
        mark42: {
            primary: 0xfbbf24,      // Gold armor
            secondary: 0xb91c1c,    // Red trim
            energy: 0x38bdf8,       // Arc reactor blue
            glow: 0x0ea5e9,
            metal: 0xd97706
        },
        war_machine: {
            primary: 0x374151,      // Gunmetal gray
            secondary: 0x1f2937,    // Dark gray
            energy: 0xef4444,       // Red energy
            glow: 0xf87171,
            metal: 0x4b5563
        },
        stealth: {
            primary: 0x1e293b,      // Dark blue-black
            secondary: 0x334155,    // Slate
            energy: 0x22d3ee,       // Cyan
            glow: 0x06b6d4,
            metal: 0x0f172a
        },
        nanotech: {
            primary: 0x7c3aed,      // Purple
            secondary: 0xa855f7,    // Light purple
            energy: 0xfbbf24,       // Gold energy
            glow: 0xf59e0b,
            metal: 0x6d28d9
        }
    }
};

// ============== GLOBAL STATE ==============
let scene, camera, renderer;
let armorSegments = [];
let repulsor, repulsorGlow;
let energyLines = [];
let powerIndicators = [];
let hudElements = [];
let handLandmarks = null;
let ws = null;
let currentTheme = CONFIG.themes.mark3;
let glowIntensity = 1;
let clock;

// Effect toggles
let effects = {
    armor: true,
    repulsor: true,
    energy: true,
    hud: true
};

// Responsive sizing
let canvasWidth, canvasHeight, aspectRatio;

// Hand landmark indices
const LANDMARKS = {
    WRIST: 0,
    THUMB: [1, 2, 3, 4],
    INDEX: [5, 6, 7, 8],
    MIDDLE: [9, 10, 11, 12],
    RING: [13, 14, 15, 16],
    PINKY: [17, 18, 19, 20]
};

// ============== INITIALIZATION ==============
function init() {
    updateCanvasSize();

    // Create scene with dark background
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a12);

    // Create camera
    camera = new THREE.OrthographicCamera(
        -aspectRatio, aspectRatio, 1, -1, 0.1, 1000
    );
    camera.position.z = 5;

    // Create renderer
    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('handCanvas'),
        antialias: true
    });
    renderer.setSize(canvasWidth, canvasHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    clock = new THREE.Clock();

    // Create all gauntlet components
    createArmorPlates();
    createRepulsor();
    createEnergyLines();
    createPowerIndicators();
    createHUDElements();
    createBackgroundGrid();

    // Setup controls
    setupControls();

    // Connect to WebSocket
    connectWebSocket();

    // Handle resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(onWindowResize, 100);
    });

    // Start animation loop
    animate();
}

function updateCanvasSize() {
    canvasWidth = window.innerWidth;
    canvasHeight = window.innerHeight;
    aspectRatio = canvasWidth / canvasHeight;
}

// ============== BACKGROUND GRID ==============
function createBackgroundGrid() {
    // Tech grid lines
    const gridMaterial = new THREE.LineBasicMaterial({
        color: 0x1e3a5f,
        transparent: true,
        opacity: 0.15
    });

    const gridSize = 4;
    const divisions = 40;

    for (let i = -divisions; i <= divisions; i++) {
        const pos = (i / divisions) * gridSize;

        // Horizontal lines
        const hGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-gridSize * aspectRatio, pos, -1),
            new THREE.Vector3(gridSize * aspectRatio, pos, -1)
        ]);
        scene.add(new THREE.Line(hGeom, gridMaterial));

        // Vertical lines
        const vGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(pos * aspectRatio, -gridSize, -1),
            new THREE.Vector3(pos * aspectRatio, gridSize, -1)
        ]);
        scene.add(new THREE.Line(vGeom, gridMaterial));
    }
}

// ============== ARMOR PLATES ==============
function createArmorPlates() {
    const fingers = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY'];

    fingers.forEach((finger, fingerIdx) => {
        const segments = LANDMARKS[finger];

        for (let i = 0; i < segments.length - 1; i++) {
            // Create armor plate geometry
            const plateGeom = new THREE.BoxGeometry(0.04, 0.025, 0.015);

            // Main plate
            const plateMat = new THREE.MeshBasicMaterial({
                color: currentTheme.primary,
                transparent: true,
                opacity: 0.9
            });
            const plate = new THREE.Mesh(plateGeom, plateMat);
            plate.visible = false;

            // Edge highlight
            const edgeGeom = new THREE.EdgesGeometry(plateGeom);
            const edgeMat = new THREE.LineBasicMaterial({
                color: currentTheme.secondary,
                transparent: true,
                opacity: 0.8
            });
            const edges = new THREE.LineSegments(edgeGeom, edgeMat);
            plate.add(edges);

            // Store metadata
            plate.userData = {
                finger: finger,
                segmentStart: segments[i],
                segmentEnd: segments[i + 1],
                fingerIndex: fingerIdx,
                segmentIndex: i
            };

            scene.add(plate);
            armorSegments.push(plate);
        }
    });

    // Palm armor (larger central plate)
    const palmGeom = new THREE.BoxGeometry(0.12, 0.1, 0.02);
    const palmMat = new THREE.MeshBasicMaterial({
        color: currentTheme.primary,
        transparent: true,
        opacity: 0.85
    });
    const palmPlate = new THREE.Mesh(palmGeom, palmMat);
    palmPlate.visible = false;
    palmPlate.userData = { isPalm: true };

    const palmEdgeGeom = new THREE.EdgesGeometry(palmGeom);
    const palmEdgeMat = new THREE.LineBasicMaterial({
        color: currentTheme.secondary,
        transparent: true,
        opacity: 0.8
    });
    palmPlate.add(new THREE.LineSegments(palmEdgeGeom, palmEdgeMat));

    scene.add(palmPlate);
    armorSegments.push(palmPlate);
}

// ============== REPULSOR ==============
function createRepulsor() {
    const group = new THREE.Group();

    // Core circle
    const coreGeom = new THREE.CircleGeometry(0.035, 32);
    const coreMat = new THREE.MeshBasicMaterial({
        color: currentTheme.energy,
        transparent: true,
        opacity: 0.95
    });
    const core = new THREE.Mesh(coreGeom, coreMat);
    group.add(core);

    // Outer rings
    for (let i = 1; i <= 3; i++) {
        const ringGeom = new THREE.RingGeometry(0.035 + i * 0.012, 0.038 + i * 0.012, 32);
        const ringMat = new THREE.MeshBasicMaterial({
            color: currentTheme.glow,
            transparent: true,
            opacity: 0.6 / i,
            side: THREE.DoubleSide
        });
        const ring = new THREE.Mesh(ringGeom, ringMat);
        group.add(ring);
    }

    // Glow particles
    const particleCount = 30;
    const particleGeom = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
        const angle = (i / particleCount) * Math.PI * 2;
        const r = 0.04 + Math.random() * 0.02;
        positions[i * 3] = Math.cos(angle) * r;
        positions[i * 3 + 1] = Math.sin(angle) * r;
        positions[i * 3 + 2] = 0;
    }

    particleGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const particleMat = new THREE.PointsMaterial({
        color: currentTheme.energy,
        size: 0.008,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending
    });

    repulsorGlow = new THREE.Points(particleGeom, particleMat);
    group.add(repulsorGlow);

    group.visible = false;
    repulsor = group;
    scene.add(group);
}

// ============== ENERGY LINES ==============
function createEnergyLines() {
    const fingers = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY'];

    fingers.forEach((finger) => {
        const segments = LANDMARKS[finger];

        // Create line from wrist to fingertip
        const points = [];
        for (let i = 0; i < segments.length; i++) {
            points.push(new THREE.Vector3(0, 0, 0));
        }

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: currentTheme.energy,
            transparent: true,
            opacity: 0.7,
            linewidth: 2
        });

        const line = new THREE.Line(geometry, material);
        line.visible = false;
        line.userData = { finger, segments };

        scene.add(line);
        energyLines.push(line);
    });
}

// ============== POWER INDICATORS ==============
function createPowerIndicators() {
    // Create small power nodes at each knuckle
    const fingers = ['INDEX', 'MIDDLE', 'RING', 'PINKY'];

    fingers.forEach((finger, idx) => {
        const baseIdx = LANDMARKS[finger][0];  // MCP joint

        const nodeGeom = new THREE.CircleGeometry(0.008, 16);
        const nodeMat = new THREE.MeshBasicMaterial({
            color: currentTheme.energy,
            transparent: true,
            opacity: 0.9
        });

        const node = new THREE.Mesh(nodeGeom, nodeMat);
        node.visible = false;
        node.userData = { landmarkIndex: baseIdx, fingerIndex: idx };

        // Outer glow ring
        const glowGeom = new THREE.RingGeometry(0.01, 0.015, 16);
        const glowMat = new THREE.MeshBasicMaterial({
            color: currentTheme.glow,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide
        });
        const glow = new THREE.Mesh(glowGeom, glowMat);
        node.add(glow);

        scene.add(node);
        powerIndicators.push(node);
    });
}

// ============== HUD ELEMENTS ==============
function createHUDElements() {
    // Targeting reticle around index finger
    const reticleGroup = new THREE.Group();

    // Outer brackets
    const bracketMat = new THREE.LineBasicMaterial({
        color: currentTheme.energy,
        transparent: true,
        opacity: 0.7
    });

    const bracketSize = 0.06;
    const bracketOffset = 0.04;

    // Corner brackets
    const corners = [
        { x: -1, y: 1 }, { x: 1, y: 1 },
        { x: -1, y: -1 }, { x: 1, y: -1 }
    ];

    corners.forEach(corner => {
        const points = [
            new THREE.Vector3(corner.x * bracketOffset, corner.y * bracketSize, 0),
            new THREE.Vector3(corner.x * bracketOffset, corner.y * bracketOffset, 0),
            new THREE.Vector3(corner.x * bracketSize, corner.y * bracketOffset, 0)
        ];
        const geom = new THREE.BufferGeometry().setFromPoints(points);
        const bracket = new THREE.Line(geom, bracketMat);
        reticleGroup.add(bracket);
    });

    // Center crosshair
    const crossSize = 0.015;
    const crossGeom1 = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-crossSize, 0, 0),
        new THREE.Vector3(crossSize, 0, 0)
    ]);
    const crossGeom2 = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -crossSize, 0),
        new THREE.Vector3(0, crossSize, 0)
    ]);
    reticleGroup.add(new THREE.Line(crossGeom1, bracketMat));
    reticleGroup.add(new THREE.Line(crossGeom2, bracketMat));

    reticleGroup.visible = false;
    reticleGroup.userData = { type: 'reticle' };
    scene.add(reticleGroup);
    hudElements.push(reticleGroup);

    // Power level arc
    const arcGroup = new THREE.Group();
    const arcGeom = new THREE.RingGeometry(0.08, 0.085, 32, 1, 0, Math.PI * 1.5);
    const arcMat = new THREE.MeshBasicMaterial({
        color: currentTheme.energy,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide
    });
    const arc = new THREE.Mesh(arcGeom, arcMat);
    arcGroup.add(arc);
    arcGroup.visible = false;
    arcGroup.userData = { type: 'powerArc' };
    scene.add(arcGroup);
    hudElements.push(arcGroup);
}

// ============== UPDATE FUNCTIONS ==============
function updateArmorPlates(time) {
    if (!handLandmarks || !effects.armor) {
        armorSegments.forEach(plate => plate.visible = false);
        return;
    }

    const landmarks = handLandmarks.landmarks;

    armorSegments.forEach(plate => {
        if (plate.userData.isPalm) {
            // Position palm plate at center of palm
            const palmCenter = handLandmarks.palm;
            const x = (palmCenter.x - 0.5) * 2 * aspectRatio;
            const y = -(palmCenter.y - 0.5) * 2;

            plate.position.set(x, y, 0.01);
            plate.visible = true;

            // Subtle pulse
            const pulse = 1 + Math.sin(time * 2) * 0.03 * glowIntensity;
            plate.scale.setScalar(pulse);
        } else {
            // Position finger segment plates
            const startLm = landmarks[plate.userData.segmentStart];
            const endLm = landmarks[plate.userData.segmentEnd];

            if (startLm && endLm) {
                const midX = ((startLm.x + endLm.x) / 2 - 0.5) * 2 * aspectRatio;
                const midY = -((startLm.y + endLm.y) / 2 - 0.5) * 2;

                plate.position.set(midX, midY, 0.02);

                // Calculate rotation based on segment direction
                const dx = (endLm.x - startLm.x) * aspectRatio;
                const dy = -(endLm.y - startLm.y);
                const angle = Math.atan2(dy, dx);
                plate.rotation.z = angle;

                // Scale based on segment length
                const length = Math.sqrt(dx * dx + dy * dy) * 2;
                plate.scale.x = Math.max(0.5, length * 1.5);

                plate.visible = true;
            }
        }
    });
}

function updateRepulsor(time) {
    if (!handLandmarks || !effects.repulsor) {
        repulsor.visible = false;
        return;
    }

    repulsor.visible = true;
    const palm = handLandmarks.palm;

    const x = (palm.x - 0.5) * 2 * aspectRatio;
    const y = -(palm.y - 0.5) * 2;

    repulsor.position.set(x, y, 0.03);

    // Pulsing animation
    const pulse = 1 + Math.sin(time * 4) * 0.15 * glowIntensity;
    repulsor.scale.setScalar(pulse);

    // Rotate glow particles
    if (repulsorGlow) {
        repulsorGlow.rotation.z += 0.03;
    }

    // Update particle positions for energy effect
    const positions = repulsorGlow.geometry.attributes.position.array;
    for (let i = 0; i < positions.length / 3; i++) {
        const angle = (i / (positions.length / 3)) * Math.PI * 2 + time * 2;
        const r = 0.04 + Math.sin(time * 3 + i) * 0.01;
        positions[i * 3] = Math.cos(angle) * r;
        positions[i * 3 + 1] = Math.sin(angle) * r;
    }
    repulsorGlow.geometry.attributes.position.needsUpdate = true;
}

function updateEnergyLines(time) {
    if (!handLandmarks || !effects.energy) {
        energyLines.forEach(line => line.visible = false);
        return;
    }

    const landmarks = handLandmarks.landmarks;

    energyLines.forEach(line => {
        const segments = line.userData.segments;
        const positions = line.geometry.attributes.position.array;

        let allValid = true;
        segments.forEach((segIdx, i) => {
            const lm = landmarks[segIdx];
            if (lm) {
                positions[i * 3] = (lm.x - 0.5) * 2 * aspectRatio;
                positions[i * 3 + 1] = -(lm.y - 0.5) * 2;
                positions[i * 3 + 2] = 0.01;
            } else {
                allValid = false;
            }
        });

        line.geometry.attributes.position.needsUpdate = true;
        line.visible = allValid;

        // Pulsing opacity
        line.material.opacity = 0.5 + Math.sin(time * 3) * 0.2 * glowIntensity;
    });
}

function updatePowerIndicators(time) {
    if (!handLandmarks || !effects.energy) {
        powerIndicators.forEach(node => node.visible = false);
        return;
    }

    const landmarks = handLandmarks.landmarks;

    powerIndicators.forEach((node, idx) => {
        const lm = landmarks[node.userData.landmarkIndex];
        if (lm) {
            const x = (lm.x - 0.5) * 2 * aspectRatio;
            const y = -(lm.y - 0.5) * 2;

            node.position.set(x, y, 0.025);
            node.visible = true;

            // Staggered pulsing
            const pulse = 1 + Math.sin(time * 4 + idx * 0.5) * 0.3 * glowIntensity;
            node.scale.setScalar(pulse);
        }
    });
}

function updateHUDElements(time) {
    if (!handLandmarks || !effects.hud) {
        hudElements.forEach(el => el.visible = false);
        return;
    }

    const landmarks = handLandmarks.landmarks;
    const indexTip = landmarks[8];  // Index fingertip
    const palm = handLandmarks.palm;

    hudElements.forEach(element => {
        if (element.userData.type === 'reticle' && indexTip) {
            const x = (indexTip.x - 0.5) * 2 * aspectRatio;
            const y = -(indexTip.y - 0.5) * 2;

            element.position.set(x, y, 0.04);
            element.visible = true;

            // Slow rotation
            element.rotation.z = time * 0.5;
        }

        if (element.userData.type === 'powerArc' && palm) {
            const x = (palm.x - 0.5) * 2 * aspectRatio - 0.12;
            const y = -(palm.y - 0.5) * 2;

            element.position.set(x, y, 0.02);
            element.visible = true;
            element.rotation.z = -Math.PI / 4;
        }
    });
}

// ============== CONTROLS ==============
function setupControls() {
    document.getElementById('toggleOrb').addEventListener('change', (e) => {
        effects.repulsor = e.target.checked;
    });
    document.getElementById('toggleTrails').addEventListener('change', (e) => {
        effects.energy = e.target.checked;
    });
    document.getElementById('toggleRings').addEventListener('change', (e) => {
        effects.armor = e.target.checked;
    });
    document.getElementById('toggleAura').addEventListener('change', (e) => {
        effects.hud = e.target.checked;
    });

    document.getElementById('glowIntensity').addEventListener('input', (e) => {
        glowIntensity = parseFloat(e.target.value);
    });

    document.getElementById('colorTheme').addEventListener('change', (e) => {
        currentTheme = CONFIG.themes[e.target.value];
        updateThemeColors();
    });
}

function updateThemeColors() {
    // Update armor plates
    armorSegments.forEach(plate => {
        plate.material.color.setHex(currentTheme.primary);
        if (plate.children[0]) {
            plate.children[0].material.color.setHex(currentTheme.secondary);
        }
    });

    // Update repulsor
    repulsor.children.forEach((child, i) => {
        if (i === 0) {
            child.material.color.setHex(currentTheme.energy);
        } else if (child.material) {
            child.material.color.setHex(currentTheme.glow);
        }
    });
    if (repulsorGlow) {
        repulsorGlow.material.color.setHex(currentTheme.energy);
    }

    // Update energy lines
    energyLines.forEach(line => {
        line.material.color.setHex(currentTheme.energy);
    });

    // Update power indicators
    powerIndicators.forEach(node => {
        node.material.color.setHex(currentTheme.energy);
        if (node.children[0]) {
            node.children[0].material.color.setHex(currentTheme.glow);
        }
    });

    // Update HUD
    hudElements.forEach(el => {
        el.traverse(child => {
            if (child.material) {
                child.material.color.setHex(currentTheme.energy);
            }
        });
    });
}

// ============== WEBSOCKET ==============
function connectWebSocket() {
    const status = document.querySelector('#status span');
    const statusDot = document.querySelector('.status-dot');

    ws = new WebSocket(CONFIG.websocketUrl);

    ws.onopen = () => {
        status.textContent = 'JARVIS Online - Show your hand!';
        statusDot.classList.add('connected');
    };

    ws.onmessage = (event) => {
        handLandmarks = JSON.parse(event.data);
        status.textContent = 'Gauntlet Active';
        statusDot.classList.remove('connected');
        statusDot.classList.add('tracking');
        document.getElementById('instructions').classList.add('hidden');
    };

    ws.onclose = () => {
        status.textContent = 'Reconnecting to JARVIS...';
        statusDot.classList.remove('connected', 'tracking');
        handLandmarks = null;
        document.getElementById('instructions').classList.remove('hidden');
        setTimeout(connectWebSocket, CONFIG.reconnectInterval);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// ============== ANIMATION LOOP ==============
function animate() {
    requestAnimationFrame(animate);

    const time = clock.getElapsedTime();

    updateArmorPlates(time);
    updateRepulsor(time);
    updateEnergyLines(time);
    updatePowerIndicators(time);
    updateHUDElements(time);

    renderer.render(scene, camera);
}

// ============== RESPONSIVE ==============
function onWindowResize() {
    updateCanvasSize();

    camera.left = -aspectRatio;
    camera.right = aspectRatio;
    camera.updateProjectionMatrix();

    renderer.setSize(canvasWidth, canvasHeight);
}

// ============== START ==============
init();
