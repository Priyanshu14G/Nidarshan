// Load the model
// Load the model
let loadedModel;

async function loadModel() {
  console.log("AAgya")
  loadedModel = await tf.loadLayersModel('./model.json');
}
// Function to preprocess and predict
async function predict() {
  
  await loadModel();
  console.log(loadedModel); 
  const fileInput = document.getElementById('file-input');
  const previewImg = document.getElementById('preview_img');


  const loader = document.getElementById('loader');
  loader.classList.remove("hidden");

  if (fileInput.files.length > 0) {
    const file = fileInput.files[0];
    
    // Create a URL for the selected image
    const imageURL = URL.createObjectURL(file);
    
    // Display the selected image in the 'preview_img' element
    previewImg.src = imageURL;
    
    // Load the image and preprocess it
    const imgElement = new Image();
    imgElement.src = imageURL;
    
    imgElement.onload = async () => {
      const imgTensor = tf.browser.fromPixels(imgElement);
      const resizedImg = tf.image.resizeBilinear(imgTensor, [224, 224]);
      const x = resizedImg.toFloat().div(tf.scalar(255));
      const xBatch = x.expandDims(0);

      // Make predictions
      const preds = loadedModel.predict(xBatch);
      const predArray = await preds.data();

      // Find the class with the highest probability
      const maxIndex = predArray.indexOf(Math.max(...predArray));

      // Map the index to your class labels
      const classLabels = [
        "Chickenpox",
        "Measles",
        "Monkeypox",
        "Normal"
      ];

      const prediction = classLabels[maxIndex];

      console.log('Prediction:', prediction);

      const resultdiv1=document.getElementById("result-here");
      const resultdiv=document.getElementById("result");
      resultdiv.innerText=prediction;
      resultdiv1.classList.remove("hidden");

      const loader = document.getElementById('loader');
      loader.classList.add("hidden");



    };
  }
}

// Function to handle form submission
function seeResults(event) {
  event.preventDefault();
  console.log("Hello");
  predict();
}


// Function to load a preview of the selected image
var loadFile = function (event) {
  var input = event.target;
  var file = input.files[0];
  var type = file.type;
  var getter = document.getElementById('dummy');
  getter.classList.add("hidden");

  var output = document.getElementById('preview_img');
  output.classList.remove("hidden");

  output.src = URL.createObjectURL(event.target.files[0]);
  output.onload = function () {
    URL.revokeObjectURL(output.src); // free memory
  };
};