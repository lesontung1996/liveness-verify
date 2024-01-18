
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver } = vision;

const questionList = {
  blink: "Blink your eyes",
  up: "Head up",
  down: "Head down",
  left: "Turn your face left",
  right: "Turn your face right",
}

const headposes = {
  up: "Up",
  down: "Down",
  left: "Left",
  right: "Right",
  forward: "Forward",
  blink: "Blink",
}

const questionStatuses = {
  passed: "passed",
  failed: "failed",
}

let faceLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let mediaStream;

let randomQuestionList
let testStarted = false
let roll = 0, pitch = 0, yaw = 0;
let x, y, z;
let headPose = headposes.forward
let headInFrame = false
let verifyMode

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1,
        selfieMode: true
    });
    showStep(1)
};
createFaceLandmarker();

function loadOpenCV() {
  if (typeof cv === 'undefined') {
    setTimeout(loadOpenCV, 100)
    return
  }
  cv().then((result) => {
    cv = result
  })
}
loadOpenCV()

function loadTailwind() {
  if (typeof tailwind === 'undefined') {
    setTimeout(loadTailwind, 100)
    return
  }
  document.body.classList.remove('hidden')
}
loadTailwind()

window.addEventListener("resize", setVideoDimension);

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/
const video = document.getElementById("webcam");
const webcamOverlays = document.querySelectorAll(".js-webcam-overlay");
const canvasElement = document.getElementById("canvas");
const instructionElement = document.getElementById("instruction");
const alertSuccessElement = document.getElementById("alert-success");
const alertErrorElement = document.getElementById("alert-error");
const canvasCtx = canvasElement.getContext("2d", { willReadFrequently: true });

// Check if webcam access is supported.
function hasGetUserMedia() {
    return !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.mediaDevices.getUserMedia);
}
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    const startLivenessTestButtons = document.querySelectorAll(".js-start-liveness");
    const startDocumentVerifytButtons = document.querySelectorAll(".js-start-document");
    const startAddressVerifytButtons = document.querySelectorAll(".js-start-address");
    const continueVerifyDocumenttButtons = document.querySelectorAll(".js-continue-document");
    const showButtons = document.querySelectorAll("[class*=js-show]");
    const captureButtons = document.querySelectorAll('.js-capture-camera')
    captureButtons.forEach(button => {
      button.addEventListener('click', captureFrame)
    })
    startLivenessTestButtons.forEach(button => {
      button.addEventListener("click", enableCameraForLiveness);
    })
    startDocumentVerifytButtons.forEach(button => {
      button.addEventListener("click", enableCameraForDocumentVerify);
    })
    continueVerifyDocumenttButtons.forEach(button => {
      button.addEventListener("click", continueVerifyDocument);
    })
    startAddressVerifytButtons.forEach(button => {
      button.addEventListener("click", enableCameraForAddressVerify);
    })
    showButtons.forEach(button => {
      const className = [...button.classList].find(className => className.includes('js-show'))
      if (!className) return
      const stepName = className.replace('js-show-', '')
      button.addEventListener("click", () => showStep(stepName));
    })
} else {
    alert("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
function enableCameraForLiveness(event) {
  showStep('loading')
    if (!faceLandmarker || !cv) {
      alert("Wait! faceLandmarker and OpenCV not loaded yet.");
      return;
    }
    verifyMode = 'liveness'
    webcamRunning = true;
    // getUsermedia parameters.
    const constraints = {
      audio: false,
      video: {
        facingMode: 'user'
      }
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      mediaStream = stream
      video.srcObject = stream;
      video.addEventListener("loadeddata", () => {
        showStep('verify')
        predictCameraForLivenes()
      });
      startLivenessTest()
    });
}
let lastVideoTime = -1;
let results = undefined;
async function predictCameraForLivenes() {
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
      runningMode = "VIDEO";
      await faceLandmarker.setOptions({ runningMode: runningMode });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime;
      results = faceLandmarker.detectForVideo(video, startTimeMs);
  }
  if (results.faceLandmarks) {
    try {
      calculateHeadPose(results)
    } catch (error) {
      console.warn(error)
    }
  }
  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
      window.requestAnimationFrame(predictCameraForLivenes);
  }
}

function setVideoDimension() {
  if (!video.offsetWidth || !video.offsetHeight) return
  let actualWidth, actualHeight
  const videoWidth = video.offsetWidth;
  const videoHeight = video.offsetHeight;
  const ratio = video.videoHeight / video.videoWidth;
  const windowRatio = window.innerHeight / window.innerWidth
  if (ratio >= windowRatio) {
    actualWidth = videoHeight / ratio
    actualHeight = videoHeight
  } else {
    actualWidth = videoWidth
    actualHeight = videoWidth * ratio
  }
  canvasElement.width = actualWidth;
  canvasElement.height = actualHeight;
  webcamOverlays.forEach(webcamOverlay => {
    const webcamOverlayOvals = webcamOverlay.querySelectorAll(".overlay-ellipse");
    const webcamOverlayRects = webcamOverlay.querySelectorAll(".overlay-rect");
    webcamOverlay.setAttribute("viewBox", `0 0 ${window.innerWidth} ${window.innerHeight}`)

    webcamOverlayOvals.forEach(item => {
      item.setAttribute("rx", ratio > 1 ? actualWidth * 0.35 : actualHeight * 0.8 * 0.35)
      item.setAttribute("ry", ratio > 1 ? actualWidth / 0.8 * 0.35 : actualHeight * 0.35)
    })

    webcamOverlayRects.forEach(item => {
      const overlayWidth = ratio > 1 ? actualWidth * 0.9 : actualWidth * 0.7
      const overlayHeight = ratio > 1 ? actualHeight * 0.5 : actualHeight * 0.65
      item.setAttribute("width", overlayWidth)
      item.setAttribute("height", overlayHeight)
      item.setAttribute("transform", `translate(-${overlayWidth / 2},-${overlayHeight / 2})`)
    })
  })
}

function calculateHeadPose(results) {

  var face_2d = [];
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj      
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
  var points = [1, 33, 263, 61, 291, 199];
  var pointsObj = [0, -1.126865, 7.475604,
      -4.445859, 2.663991, 3.173422,
      4.445859, 2.663991, 3.173422,
      -2.456206, -4.342621, 4.283884,
      2.456206, -4.342621, 4.283884,
      0, -9.403378, 4.264492]; //chin
  // Draw the overlays
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  var width = canvasElement.width; //canvasElement.width; //
  var height = canvasElement.height; //canvasElement.height; //results.image.height;
  // Camera internals
  var normalizedFocaleY = 1.28; // Logitech 922
  var focalLength = height * normalizedFocaleY;
  var s = 0; //0.953571;
  var cx = width / 2;
  var cy = height / 2;
  var cam_matrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
      focalLength,
      s,
      cx,
      0,
      focalLength,
      cy,
      0,
      0,
      1
  ]);
  //The distortion parameters
  //var dist_matrix = cv.Mat.zeros(4, 1, cv.CV_64FC1); // Assuming no lens distortion
  var k1 = 0.1318020374;
  var k2 = -0.1550007612;
  var p1 = -0.0071350401;
  var p2 = -0.0096747708;
  var dist_matrix = cv.matFromArray(4, 1, cv.CV_64FC1, [k1, k2, p1, p2]);
  headInFrame = calculateHeadInFrame(results)

  if (results.faceLandmarks) {
    for (const landmarks of results.faceLandmarks) {
      for (const point of points) {
        var point0 = landmarks[point];
        const x = point0.x * width;
        const y = point0.y * height;
        //var z = point0.z; 
        // Get the 2D Coordinates
        face_2d.push(x);
        face_2d.push(y);
      }
    }
  }
  if (face_2d.length > 0) {
    // Initial guess
    //Rotation in axis-angle form
    var rvec = new cv.Mat(); // = cv.matFromArray(1, 3, cv.CV_64FC1, [0, 0, 0]); //new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1); // Output rotation vector
    var tvec = new cv.Mat(); // = cv.matFromArray(1, 3, cv.CV_64FC1, [-100, 100, 1000]); //new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1); // Output translation vector
    const numRows = points.length;
    const imagePoints = cv.matFromArray(numRows, 2, cv.CV_64FC1, face_2d);
    var modelPointsObj = cv.matFromArray(6, 3, cv.CV_64FC1, pointsObj);
    // https://docs.opencv.org/4.6.0/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
    // https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html
    var success = cv.solvePnP(modelPointsObj, //modelPoints,
    imagePoints, cam_matrix, dist_matrix, rvec, // Output rotation vector
    tvec, false, //  uses the provided rvec and tvec values as initial approximations
    cv.SOLVEPNP_ITERATIVE //SOLVEPNP_EPNP //SOLVEPNP_ITERATIVE (default but pose seems unstable)
    );
    if (success) {
      var rmat = cv.Mat.zeros(3, 3, cv.CV_64FC1);
      const jaco = new cv.Mat();
      // Get rotational matrix rmat
      cv.Rodrigues(rvec, rmat, jaco); // jacobian	Optional output Jacobian matrix
      var sy = Math.sqrt(rmat.data64F[0] * rmat.data64F[0] + rmat.data64F[3] * rmat.data64F[3]);
      var singular = sy < 1e-6;
      // we need decomposeProjectionMatrix
      if (!singular) {
          x = Math.atan2(rmat.data64F[7], rmat.data64F[8]);
          y = Math.atan2(-rmat.data64F[6], sy);
          z = Math.atan2(rmat.data64F[3], rmat.data64F[0]);
      }
      else {
          x = Math.atan2(-rmat.data64F[5], rmat.data64F[4]);
          y = Math.atan2(-rmat.data64F[6], sy);
          z = 0;
      }
      roll = (180.0 * (z / Math.PI));
      pitch = (180.0 * (x / Math.PI));
      yaw = (180.0 * (y / Math.PI));

      const eyeBlinkLeft = results.faceBlendshapes[0].categories.find(item => item.categoryName === 'eyeBlinkLeft').score
      const eyeBlinkRight = results.faceBlendshapes[0].categories.find(item => item.categoryName === 'eyeBlinkRight').score

      if (yaw < -20) {
        headPose = headposes.left
      } else if (yaw > 20) {
        headPose = headposes.right
      } else if (pitch > 0 && Math.abs(pitch) < 165) {
        headPose = headposes.up
      } else if (pitch < 0 && Math.abs(pitch) < 165) {
        headPose = headposes.down
      } else if (eyeBlinkLeft > 0.4 && eyeBlinkRight > 0.4) {
        headPose = headposes.blink
      } else {
        headPose = headposes.forward
      }
    }
    rvec.delete();
    tvec.delete();
  }
  canvasCtx.restore();
}

// =======================

function calculateHeadInFrame(results) {
  if (!results.faceLandmarks[0]) return

  const landmarks = results.faceLandmarks[0]
  const points = [1, 33, 263, 61, 291, 199]
  const videoRatio = video.videoHeight / video.videoWidth

  return points.every(point => {
    const point0 = landmarks[point]
    if (videoRatio > 1) {
      return point0.x >= 0.2 && point0.x <= 0.8 && point0.y >= 0.2 && point0.y <= 0.8
    } else {
      return point0.x >= 0.3 && point0.x <= 0.7 && point0.y >= 0.15 && point0.y <= 0.85
    }
  })
}

function startLivenessTest () {
  if (testStarted === true) {
    return
  }
  randomQuestionList = generateQuestionList()

  startTestHeadInFrame()
  testStarted = true
}

function startTestHeadInFrame() {
  let currentFrames = 0
  const targetFrames = 10
  instructionElement.textContent = `Keep your face within the oval to start recording`

  const currentInterval = setInterval(() => {
    if (headInFrame === true) {
      currentFrames = currentFrames + 1
      if (currentFrames >= targetFrames) {
        showAlert()
        setTimeout(() => startQuestion(0), 2000)
        clearInterval(currentInterval)
        currentFrames = 0
      }
    } else {
      currentFrames = 0
    }
  })
}

function startQuestion (currentStep) {
  const questionObject = randomQuestionList[currentStep]
  const questionKey = questionObject.key
  instructionElement.textContent = `${questionObject.text}`

  const currentInterval = setInterval(() => {
    if (questionKey === 'blink') {
      if (headposes[questionKey] !== headPose) return
    } else {
      if (headPose === headposes.forward || headPose === headposes.blink) return
      if (headposes[questionKey] !== headPose) {
        showFail()
        clearInterval(currentInterval)
        return
      }
    }
    
    showSuccess(currentStep)
    clearInterval(currentInterval)
  }, 100)
}

function generateQuestionList() {
  const randomizedList = Object.keys(questionList).map(key => ({
    value: {
      key,
      text: questionList[key],
    },
    sort: Math.random()
  }))
  .sort((a, b) => a.sort - b.sort)
  .map(({ value }) => value)

  const randomItem = getOneRandomQuestion()

  randomizedList.push(randomItem)

  return randomizedList
}

function getOneRandomQuestion() {
  const keys = Object.keys(questionList)
  const randomKey = keys[keys.length * Math.random() << 0]
  return {
    key: randomKey,
    text: questionList[randomKey]
  }
}

function showFail() {
  showAlert('error')
  setTimeout(() => {
    showStep('fail')
    stopCamera()
  }, 2000)
}

function showSuccess(currentStep) {
  const nextQuestionStep = currentStep + 1

  showAlert()
  setTimeout(() => {
    if (currentStep === randomQuestionList.length - 1) {
      showStep('success')
      stopCamera()
    } else {
      startQuestion(nextQuestionStep)
    }
  }, 2000)
}

function stopCamera() {
  if (testStarted === false) return
  webcamRunning = false
  testStarted = false
  mediaStream.getTracks().forEach(function(track) {
    track.stop()
  })
}

function showStep(name, additionalClass = null) {
  const steps = document.querySelectorAll('.step')

  const target = Number.isInteger(name) ? steps[name] : document.getElementById(`step-${name}`) 
  if (!target) return
  steps.forEach(step => step.classList.add('hidden'))
  target.classList.remove('hidden')

  if (name === 'verify') {
    const toDeletes = [...target.classList].filter(className => className.includes('verify--'))
    toDeletes.forEach(className => target.classList.remove(className))
    target.classList.add(`verify--${verifyMode}`)
    setVideoDimension()
    setVideoInstruction()
  }
}

function setVideoInstruction() {
  let instruction
  if (verifyMode === 'document') {
    const questionObject = documenQuestiontList[currentDocumentStep]
    instruction = questionObject.text
  } else if (verifyMode === 'address') {
    instruction = 'Capture proof of address'
  } else {
    return
  }
  instructionElement.textContent = instruction
}

function showAlert(type = 'success') {
  const alertElement = type === 'success' ? alertSuccessElement : alertErrorElement
  const stepContainer = alertElement.closest('.step')

  alertElement.classList.remove('hidden')
  stepContainer.classList.add(`step--${type}`)
  setTimeout(() => {
    alertElement.classList.add('hidden')
    stepContainer.classList.remove(`step--${type}`)
  }, 2000)
}


// ==================== End Liveness Verification


// ==================== Start Document Verification

var createWorker, createScheduler, scheduler
const scanner = new jscanify()

async function loadTesseract() {
  if (typeof Tesseract === 'undefined') {
    setTimeout(loadTesseract, 100)
  }
  createWorker = Tesseract.createWorker
  createScheduler = Tesseract.createScheduler
  scheduler = createScheduler()
  const worker = await createWorker();
  scheduler.addWorker(worker);
}
loadTesseract()

const documenQuestiontList = [
  {
    key: 'front',
    text: "Capture a pic of your ID's front side in the frame",
    title: "ID Front side"
  },
  {
    key: 'back',
    text: "Capture a pic of your ID's back side in the frame",
    title: "ID Back side"
  }
]

let currentDocumentStep = 0

function enableCameraForDocumentVerify() {
  showStep('loading')
  if (typeof Tesseract === 'undefined' || typeof cv === 'undefined') {
    setTimeout(enableCameraForDocumentVerify, 100)
    return;
  }
  verifyMode = 'document'
  webcamRunning = true;
  // getUsermedia parameters.
  const constraints = {
    audio: false,
    video: {
      facingMode: 'environment',
      height: { min: 1200 },
    }
  };
  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    mediaStream = stream
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
      showStep('verify')
    });
  });
}

function updateSuccessDocumentPage(canvas) {
  // Create combinedCanvas to send to backend
  // const combinedCanvas = document.getElementById('combinedCanvas');
  // const combinedContext = combinedCanvas.getContext('2d', { willReadFrequently: true });

  // documenQuestiontList.forEach((item, index) => {
  //   const canvas = item.result
  //   if (index === 0) {
  //     combinedCanvas.style.width = canvas.width
  //     combinedCanvas.style.height = canvas.height * documenQuestiontList.length
  //     combinedCanvas.width = canvas.width
  //     combinedCanvas.height = canvas.height * documenQuestiontList.length
  //   }
  //   combinedContext.drawImage(canvas, 0, canvas.height * index);
  // })
  const questionObject = documenQuestiontList[currentDocumentStep]
  const container = document.querySelector("#document-result")
  const title = container.parentNode.querySelector("h1")
  while (container.firstChild) container.removeChild(container.firstChild)

  title.textContent = questionObject.title
  const image = document.createElement("img")
  image.src = canvas.toDataURL()
  image.className = 'block w-10/12 rounded-lg'
  container.appendChild(image)
}

function continueVerifyDocument() {
  if (currentDocumentStep === documenQuestiontList.length - 1) {
    showStep('welcome-liveness')
  } else {
    currentDocumentStep = currentDocumentStep + 1
    enableCameraForDocumentVerify()
  }
}

// =======================

function enableCameraForAddressVerify() {
  showStep('loading')
  verifyMode = 'address'
  webcamRunning = true
  testStarted = true
  // getUsermedia parameters.
  const constraints = {
    audio: false,
    video: {
      facingMode: 'environment',
      width: { ideal: 1280 },
    }
  };
  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    mediaStream = stream
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
      showStep('verify')
    });
  });
}

function captureFrame() {
  const canvas = getCanvasFromVideo()

  if (verifyMode === 'document') {
    const extractedCanvas = scanner.extractPaper(canvas, 1000, 630)
    updateSuccessDocumentPage(extractedCanvas)
    showStep('success-document')
  } else if (verifyMode === 'address') {
    updateSuccessAddressPage(canvas)
    showStep('success-address')
  }

  stopCamera()
}

function updateSuccessAddressPage(canvas) {
  if (!canvas) return
  const container = document.querySelector("#address-result")
  while (container.firstChild) container.removeChild(container.firstChild)

  const image = document.createElement("img")
  image.src = canvas.toDataURL()
  image.className = 'block rounded-lg'
  container.appendChild(image)
}

function getCanvasFromVideo(videoEl = undefined) {
  if (!videoEl) {
    videoEl = video
  }
  const canvas = document.createElement('canvas');
  const canvasCtx = canvas.getContext('2d', { willReadFrequently: true })
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  canvasCtx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  return canvas
}
