
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
import { countryList, mockApplicant, countryObject } from "./statics.js";

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
const store = {
  applicantName: '',
  applicantId: '',
  clientId: 'ad969161-95a0-4df3-a6e7-83e31fcb250e',
  formDocument: {
    id: '',
    type: '',
    country: '',
  },
  formAddress: {
    type: '',
    country: '',
  },
  resultDocument: {
    canvasFront: null,
    canvasBack: null,
  },
  resultLiveness: {
    canvasFace_1: null
  },
  resultAddress: {
    canvasAddress: null
  },
  apiResponseDocument: null,
  apiResponseLiveness: null,
  apiResponseAddress: null,
  apiResponseFinish: null,
  apiResponseApplicant: null,
}

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

function renderContry() {
  const selectEls = document.querySelectorAll('.js-select-country')
  selectEls.forEach(selectElement => {
      // Create and append new options
      countryList.forEach(country => {
        const optionElement = document.createElement('option');
        optionElement.value = country.value;
        optionElement.text = country.text;
        selectElement.appendChild(optionElement);
      });
  })
}
renderContry()

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
    const continueVerifyAddressButtons = document.querySelectorAll(".js-continue-address");
    const continueFinishButtons = document.querySelectorAll(".js-continue-finish");
    const showButtons = document.querySelectorAll("[class*=js-show]");
    const forms = document.querySelectorAll("form[id*=form]");
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
    continueVerifyAddressButtons.forEach(button => {
      button.addEventListener("click", apiAddress);
    })
    startAddressVerifytButtons.forEach(button => {
      button.addEventListener("click", enableCameraForAddressVerify);
    })
    continueFinishButtons.forEach(button => {
      button.addEventListener("click", apiRequestVerify);
    })
    showButtons.forEach(button => {
      const className = [...button.classList].find(className => className.includes('js-show'))
      if (!className) return
      const stepName = className.replace('js-show-', '')

      button.addEventListener("click", () => showStep(stepName));
    })
    forms.forEach(form => {
      form.addEventListener('submit', async (e) => {
        if (form.id === 'form-password') {
          e.preventDefault()
          const input = form.querySelector('input')
          if (input.value !== 'passport') {
            alert('Wrong password')
          } else {
            showStep('applicant')
          }
        }

        if (form.id === 'form-applicant') {
          e.preventDefault()
          showStep('loading')
          const input = form.querySelector('input')
          const myHeaders = new Headers();
          myHeaders.append("x-client-id", store.clientId);
          myHeaders.append("Content-Type", "application/json");

          const raw = JSON.stringify({
            full_name: input.value
          });
          try {
            const response = await fetch("https://develop.kyc.passport.stuffio.com/kyc/applicants", {
              method: 'POST',
              headers: myHeaders,
              body: raw,
              redirect: 'follow'
            });
            const responseJson = await response.json();
            store.applicantName = responseJson.full_name
            store.applicantId = responseJson.applicant_id
            showStep('welcome')
          } catch (error) {
            handleApiError(error)
          }
          
        }

        if (form.getAttribute('id') === 'form-identity') {
          e.preventDefault()
          const inputId = form.querySelector('input[name=id]')
          const selectDocumentType = form.querySelector('select[name=document-type]')
          const selectCountry = form.querySelector('select[name=country]')
          
          store.formDocument.id = inputId?.value
          store.formDocument.type = selectDocumentType?.value
          store.formDocument.country = selectCountry?.value
          showStep('welcome-document-verify')
        }

        if (form.getAttribute('id') === 'form-address') {
          e.preventDefault()
          const selectDocumentType = form.querySelector('select[name=document-type]')
          const selectCountry = form.querySelector('select[name=country]')
          
          store.formAddress.type = selectDocumentType.value
          store.formAddress.country = selectCountry.value
          showStep('proof-address-2')
        }
      })
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
      }, { once: true });
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
      if (headPose === headposes.forward) {
        const canvas = getCanvasFromVideo()
        store.resultLiveness[`canvasFace_${currentFrames}`] = canvas
      }
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
      apiLiveness()
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

function apiLiveness() {
  let file
  for (let index = 0; index < Object.keys(store.resultLiveness).length; index++) {
    try {
      const key = Object.keys(store.resultLiveness)[index];
      const canvas = store.resultLiveness[key]
      file = dataURLtoFile(canvas.toDataURL(), `face.png`)
      if (typeof file.name === "string") {
        store.resultLiveness.canvasFace = canvas
        break;
      }
    } catch (error) {
      console.log(error)      
    }
  }

  const myHeaders = new Headers();
  myHeaders.append("X-Client-Id", store.clientId);
  
  const formdata = new FormData();
  formdata.append("face", file, "face.png");
  
  const requestOptions = {
    method: 'POST',
    headers: myHeaders,
    body: formdata,
    redirect: 'follow'
  };

  fetch(`https://develop.kyc.passport.stuffio.com/kyc/applicants/${store.applicantId}/face`, requestOptions)
    .then(response => response.json())
    .then(result => {
      console.log(result)
      store.apiResponseLiveness = result
    })
    .catch(error => handleApiError(error));
}


// ==================== End Liveness Verification


// ==================== Start Document Verification

let currentDocumentStep = 0

function enableCameraForDocumentVerify() {
  showStep('loading')
  if (typeof cv === 'undefined') {
    setTimeout(enableCameraForDocumentVerify, 100)
    return;
  }
  verifyMode = 'document'
  webcamRunning = true;
  // getUsermedia parameters.
  const constraints = getEnviromentVideoConstraints()
  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    mediaStream = stream
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
      showStep('verify')
    }, { once: true });
  });
}

function updateSuccessDocumentPage(canvas) {
  const questionObject = documenQuestiontList[currentDocumentStep]
  const container = document.querySelector("#document-result")
  const title = container.parentNode.querySelector("h1")
  while (container.firstChild) container.removeChild(container.firstChild)

  if (currentDocumentStep === 0) {
    store.resultDocument.canvasFront = canvas
  } else if (currentDocumentStep === 1) {
    store.resultDocument.canvasBack = canvas
  }

  title.textContent = questionObject.title
  const image = document.createElement("img")
  image.src = canvas.toDataURL()
  image.className = 'block w-10/12 rounded-lg'
  container.appendChild(image)
}

function combineCanvasDocument() {
  const combinedCanvas = document.createElement('canvas');
  const combinedContext = combinedCanvas.getContext('2d');

  combinedCanvas.style.width = store.resultDocument.canvasFront.width
  combinedCanvas.style.height = store.resultDocument.canvasFront.height * 2
  combinedCanvas.width = store.resultDocument.canvasFront.width
  combinedCanvas.height = store.resultDocument.canvasFront.height * 2

  combinedContext.drawImage(store.resultDocument.canvasFront, 0, 0);
  combinedContext.drawImage(store.resultDocument.canvasBack, 0, combinedCanvas.height);

  return combinedCanvas
}

function continueVerifyDocument() {
  if (currentDocumentStep === documenQuestiontList.length - 1) {
    apiDocument()
    showStep('welcome-liveness')
  } else {
    currentDocumentStep = currentDocumentStep + 1
    enableCameraForDocumentVerify()
  }
}

function apiDocument() {
  const combinedCanvas = combineCanvasDocument()
  const file = dataURLtoFile(combinedCanvas.toDataURL(), `document.png`)
  const myHeaders = new Headers();
  myHeaders.append("X-Client-Id", store.clientId);

  const formdata = new FormData();
  formdata.append("document_file", file, "document.png");
  formdata.append("doc_category", "proof_of_identity");
  formdata.append("doc_type", store.formDocument.type);
  formdata.append("manual_input", `identity_number: ${store.formDocument.id}`);
  formdata.append("issuing_country", store.formDocument.country);

  const requestOptions = {
    method: 'POST',
    headers: myHeaders,
    body: formdata,
    redirect: 'follow'
  };

  fetch(`https://develop.kyc.passport.stuffio.com/kyc/documents/${store.applicantId}`, requestOptions)
    .then(response => response.json())
    .then(result => {
      console.log(result)
      store.apiResponseDocument = result
    })
    .catch(error => handleApiError(error));
}

// =======================

function enableCameraForAddressVerify() {
  showStep('loading')
  verifyMode = 'address'
  webcamRunning = true
  testStarted = true
  // getUsermedia parameters.
  const constraints = getEnviromentVideoConstraints()
  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    mediaStream = stream
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
      showStep('verify')
    }, { once: true });
  });
}

function captureFrame() {
  const canvas = getCanvasFromVideo()

  if (verifyMode === 'document') {
    let extractedCanvas
    try {
      extractedCanvas = cropCanvas(canvas)
    } catch (error) {
      extractedCanvas = canvas
      console.log(error)
    }
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

  store.resultAddress.canvasAddress = canvas
  const image = document.createElement("img")
  image.src = canvas.toDataURL()
  image.className = 'block rounded-lg'
  container.appendChild(image)
}

function apiAddress() {
  showStep('loading')
  const canvas = store.resultAddress.canvasAddress
  const file = dataURLtoFile(canvas.toDataURL(), `address.png`)
  const myHeaders = new Headers();
  myHeaders.append("X-Client-Id", store.clientId);

  const formdata = new FormData();
  formdata.append("document_file", file, "address.png");
  formdata.append("doc_category", "proof_of_address");
  formdata.append("doc_type", store.formAddress.type);
  formdata.append("issuing_country", store.formAddress.country);

  const requestOptions = {
    method: 'POST',
    headers: myHeaders,
    body: formdata,
    redirect: 'follow'
  };

  fetch(`https://develop.kyc.passport.stuffio.com/kyc/documents/${store.applicantId}`, requestOptions)
    .then(response => response.json())
    .then(result => {
      console.log(result)
      store.apiResponseAddress = result
      showStep('finish')
    })
    .catch(error => handleApiError(error));
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

function dataURLtoFile(dataurl, filename) {
  var arr = dataurl.split(','),
      mime = arr[0].match(/:(.*?);/)[1],
      bstr = atob(arr[arr.length - 1]), 
      n = bstr.length, 
      u8arr = new Uint8Array(n);
  while(n--){
      u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, {type:mime});
}

function apiRequestVerify() {
  showStep('loading')
  const myHeaders = new Headers();
  myHeaders.append("X-Client-Id", store.clientId);
  myHeaders.append("Content-Type", "application/json");

  const raw = JSON.stringify({
    document_ids: [
      store.apiResponseAddress?.document_id,
      store.apiResponseDocument?.document_id,
    ],
    face_id: store.apiResponseLiveness?.face_id
  });

  const requestOptions = {
    method: 'POST',
    headers: myHeaders,
    body: raw,
    redirect: 'follow'
  };

  fetch(`https://develop.kyc.passport.stuffio.com/kyc/verify_requests/${store.applicantId}`, requestOptions)
    .then(response => response.json())
    .then(result => {
      console.log(result)
      store.apiResponseFinish = result
      setTimeout(() => {
        apiGetApplicantInfo()
      }, 5000);
    })
    .catch(error => handleApiError(error));
}

function apiGetApplicantInfo() {
  showStep('loading')
  const myHeaders = new Headers();
  myHeaders.append("X-Client-Id", store.clientId);

  const requestOptions = {
    method: 'GET',
    headers: myHeaders,
    redirect: 'follow'
  };

  fetch(`https://develop.kyc.passport.stuffio.com/kyc/applicants/${store.applicantId}`, requestOptions)
    .then(response => response.json())
    .then(result => {
      if (result.document_full_name) {
        store.apiResponseApplicant = result || mockApplicant
        renderApplicanInfo()
        showStep('final')
      } else {
        setTimeout(() => {
          apiGetApplicantInfo()
        }, 5000);
      }
    })
    .catch(error => handleApiError(error))
}

function renderApplicanInfo() {
  const { apiResponseApplicant: data } = store
  console.log(store)
  console.log(data)
  const divElement = document.createElement('div')
  const html = `
    <div class="flex gap-4 justify-between">
      <div>
        <p class="font-bold">${data.document_full_name}</p>
        <p>Age ${data.age}</p>
        <p>Valid ID</p>
        <p class="w-1/2 text-green-500">Succeed</p>
      </div>
      <div class="w-[100px] h-[140px]">
        <img class="w-full h-full rounded object-cover" src="${store.resultLiveness.canvasFace?.toDataURL()}" >
      </div>
    </div>
    <div class="mb-8">
      <div class="flex border-b mb-8">
        <p class="px-4 py-2 border-b-2 border-black">General data</p>
      </div>
      <div>
        <div class="flex mb-4 ${data.date_of_birth ? "" : "hidden"}">
          <span class="w-1/2">Date of Birth:</span>
          <span class="w-1/2 text-green-500">${data.date_of_birth}</span>
        </div>
        <div class="flex mb-4 ${data.place_of_birth ? "" : "hidden"}">
          <span class="w-1/2">Place of Birth:</span>
          <span class="w-1/2 text-green-500">${data.place_of_birth}</span>
        </div>
        <div class="flex mb-4 ${data.identity_number ? "" : "hidden"}">
          <span class="w-1/2">ID Number:</span>
          <span class="w-1/2 text-green-500">${data.identity_number}</span>
        </div>
        <div class="flex mb-4 ${data.date_of_expiry ? "" : "hidden"}">
          <span class="w-1/2">Date of expiry:</span>
          <span class="w-1/2 text-green-500">${data.date_of_expiry}</span>
        </div>
        <div class="flex mb-4 ${data.date_of_issue ? "" : "hidden"}">
          <span class="w-1/2">Date of issue:</span>
          <span class="w-1/2 text-green-500">${data.date_of_issue}</span>
        </div>
        <div class="flex mb-4 ${data.nationality ? "" : "hidden"}">
          <span class="w-1/2">Nationality:</span>
          <span class="w-1/2 text-green-500">${countryObject[data.nationality]}</span>
        </div>
        <div class="flex mb-4 ${data.issuing_country ? "" : "hidden"}">
          <span class="w-1/2">Issuing country:</span>
          <span class="w-1/2 text-green-500">${countryObject[data.issuing_country]}</span>
        </div>
        <div class="flex mb-4 ${data.address ? "" : "hidden"}">
          <span class="w-1/2">Residential Address:</span>
          <span class="w-1/2 text-green-500">${data.address}</span>
        </div>
      </div>
    </div>
    <div class="mb-8">
      <p class="font-bold mb-4">UI Card</p>
      <div class="flex gap-4 -mx-4 px-4 overflow-auto">
        <img class="block w-10/12 rounded-lg bg-gray-300" src="${store.resultDocument.canvasFront?.toDataURL()}" >
        <img class="block w-10/12 rounded-lg bg-gray-300" src="${store.resultDocument.canvasBack?.toDataURL()}" >
      </div>
    </div>
    <div class="mb-8">
      <p class="font-bold mb-4">${store.formAddress.type}</p>
      <div class="">
        <img class="block w-full rounded-lg bg-gray-300" src="${store.resultAddress.canvasAddress?.toDataURL()}" >
      </div>
    </div>
  `
  divElement.innerHTML = html
  document.getElementById("final-result").appendChild(divElement);
}

function isIosDevice() {
  if (typeof window === `undefined` || typeof navigator === `undefined`) return false;

  return /iPhone|iPad|iPod/i.test(navigator.userAgent || navigator.vendor || (window.opera && opera.toString() === `[object Opera]`));
};

function getEnviromentVideoConstraints() {
  const constraints = {
    audio: false,
    video: {
      facingMode: 'environment'
    }
  }

  if (isIosDevice()) {
    constraints.video.width = 1600
    constraints.video.height = 1200
  } else {
    constraints.video.width = 1280
  }

  return constraints
}

function cropCanvas(canvas) {
  const { width, height } = canvas
  const ratio = height / width
  const cropX = ratio > 1 ? width * 0.05 : width * 0.15;
  const cropY = ratio > 1 ? height * 0.25 : height * 0.35 / 2;
  const cropWidth = ratio > 1 ? width * 0.9 : width * 0.7;
  const cropHeight = ratio > 1 ? height * 0.5 : height * 0.65;

  const croppedCanvas = document.createElement('canvas');
  const croppedContext = croppedCanvas.getContext('2d');

  croppedCanvas.width = cropWidth
  croppedCanvas.height = cropHeight

  croppedContext.drawImage(canvas, cropX, cropY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
  return croppedCanvas
}

function handleApiError(error) {
  console.log(error)
  alert('Unexpected error, please try again another time.')
  showStep(1)
}
