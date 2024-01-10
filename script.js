
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
    showStep('welcome-document-verify')
};
createFaceLandmarker();

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
const canvasCtx = canvasElement.getContext("2d");

// Check if webcam access is supported.
function hasGetUserMedia() {
    return !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.mediaDevices.getUserMedia);
}
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    const startLivenessTestButtons = document.querySelectorAll(".js-start-liveness");
    const startDocumentVerifytButtons = document.querySelectorAll(".js-start-document");
    startLivenessTestButtons.forEach(startLivenessTestButton => {
      startLivenessTestButton.addEventListener("click", enableCameraForLiveness);
    })
    startDocumentVerifytButtons.forEach(startDocumentVerifytButton => {
      startDocumentVerifytButton.addEventListener("click", enableCameraForDocumentVerify);
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
    if (webcamRunning === true) {
      webcamRunning = false;
    }
    else {
      webcamRunning = true;
    }
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
        showStep('verify', 'verify--liveness')
        predictCameraForLivenes()
        setVideoDimension()
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
    const webcamOverlayOval = webcamOverlay.querySelector("#overlay-ellipse");
    const webcamOverlayRect = webcamOverlay.querySelector("#overlay-rect");
    webcamOverlay.setAttribute("viewBox", `0 0 ${window.innerWidth} ${window.innerHeight}`)
    if (webcamOverlayOval) {
      webcamOverlayOval.setAttribute("rx", ratio > 1 ? actualWidth * 0.35 : actualHeight * 0.8 * 0.35)
      webcamOverlayOval.setAttribute("ry", ratio > 1 ? actualWidth / 0.8 * 0.35 : actualHeight * 0.35)
    }
    if (webcamOverlayRect) {
      const overlayWidth = ratio > 1 ? actualWidth * 0.9 : actualWidth * 0.8
      const overlayHeight = actualHeight * 0.65
      webcamOverlayRect.setAttribute("width", overlayWidth)
      webcamOverlayRect.setAttribute("height", overlayHeight)
      webcamOverlayRect.setAttribute("transform", `translate(-${overlayWidth / 2},-${overlayHeight * 0.8 / 2})`)
    }
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
  const target = document.getElementById(`step-${name}`)
  steps.forEach(step => step.classList.add('hidden'))
  target.classList.remove('hidden')

  if (additionalClass) {
    target.classList.add(additionalClass)
  }
}

function showAlert(type = 'success') {
  const alertElement = type === 'success' ? alertSuccessElement : alertErrorElement

  alertElement.classList.remove('hidden')
  setTimeout(() => {
    alertElement.classList.add('hidden')
  }, 2000)
}


// ==================== End Liveness Verification


// ==================== Start Document Verification

const { createWorker, createScheduler } = Tesseract;
const scheduler = createScheduler();
let src
let dst

const documenQuestiontList = [
  {
    key: 'front',
    text: "Capture a pic of your ID's front side in the frame"
  },
  {
    key: 'back',
    text: "Capture a pic of your ID's back side in the frame"
  }
]

function enableCameraForDocumentVerify() {
  showStep('loading')
  if (typeof Tesseract === 'undefined' || typeof cv === 'undefined') {
    alert("Wait! Tesseract not loaded yet.");
    showStep('welcome-document-verify')
    return;
  }
  if (webcamRunning === true) {
    webcamRunning = false;
  }
  else {
    webcamRunning = true;
  }
  // getUsermedia parameters.
  const constraints = {
    audio: false,
    video: {
      facingMode: 'environment'
    }
  };
  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    mediaStream = stream
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
      showStep('verify', 'verify--document')
      setVideoDimension()
      initTessaract()
    });
  });
}

async function initTessaract() {
  const worker = await createWorker();
  scheduler.addWorker(worker);
  startQuestionDocument(0)
}

async function startQuestionDocument(currentStep) {
  let currentFrame = 0
  const questionObject = documenQuestiontList[currentStep]
  instructionElement.textContent = `${questionObject.text}`

  const currentInterval = setInterval(async () => {
    const canvasTemp = document.createElement('canvas');
    canvasTemp.width = canvasElement.width;
    canvasTemp.height = canvasElement.height;
    canvasTemp.getContext('2d').drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

    const rectInFrame = checkForRectInFrame(canvasTemp)

    if (questionObject.key === 'front') {
      const idIncluded = await checkForIdNumber(canvasTemp)

      if (idIncluded && rectInFrame) {
        showSuccessDocument(currentStep)
        clearInterval(currentInterval)
      }
    } else if (questionObject.key === 'back') {
      currentFrame = rectInFrame ? currentFrame + 1 : 0
      if (currentFrame >= 2) {
        showSuccessDocument(currentStep)
        clearInterval(currentInterval)
      }
    }
  }, 500)
}

function showSuccessDocument(currentStep) {
  const nextQuestionStep = currentStep + 1

  showAlert()
  setTimeout(() => {
    if (currentStep === documenQuestiontList.length - 1) {
      showStep('success')
      stopCamera()
    } else {
      startQuestionDocument(nextQuestionStep)
    }
  }, 2000)
}

async function checkForIdNumber(canvas, idNumber = '031096004213') {
  const { data: { text } } = await scheduler.addJob('recognize', canvas);
  return text.includes(idNumber)
}

function checkForRectInFrame(canvas) {
  let result = false
  const context = canvas.getContext('2d')
  // Capture the frame from the canvas
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

  // Create an OpenCV.js matrix from the captured frame
  src = cv.matFromImageData(imageData);
  dst = src.clone();

  // Convert the image to gray scale
  cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY);

  // Apply Gaussian blur to reduce noise
  cv.GaussianBlur(src, src, new cv.Size(5, 5), 0);

  // Use Canny edge detector
  cv.Canny(src, src, 50, 200, 3, true);

  // Find contours
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  // Find the largest rectangle
  let maxArea = 0;
  let maxIndex = -1;
  for (let i = 0; i < contours.size(); ++i) {
    let area = cv.contourArea(contours.get(i));

    // Check if the contour has 4 vertices (rectangular shape)
    let vertices = new cv.Mat();
    cv.approxPolyDP(contours.get(i), vertices, 0.04 * cv.arcLength(contours.get(i), true), true);

    // Check if the area of the bounding rectangle is at least 20% of the frame
    let sizeCondition = area >= src.total() * 0.1;

    if (area > maxArea) {
      console.log(area)
      maxArea = area;
      maxIndex = i;
    }

    vertices.delete();
  }

  if (maxIndex !== -1) {
    // Get the bounding box of the largest rectangle
    let rect = cv.boundingRect(contours.get(maxIndex));

    // Draw the largest rectangle
    cv.drawContours(dst, contours, maxIndex, [0, 255, 0, 255], 2);

    // Draw the bounding box
    cv.rectangle(dst, new cv.Point(rect.x, rect.y), new cv.Point(rect.x + rect.width, rect.y + rect.height), [0, 0, 255, 255], 2);

    // Log the position (x, y) of the largest rectangle
    console.log('Position (x, y):', rect, rect.x, rect.y, maxIndex, maxArea);

    result = rect
  }

  // Display the result
  cv.imshow('canvas', dst);

  // Clean up
  contours.delete();
  hierarchy.delete();
  src.delete();
  dst.delete();
  return result
}
