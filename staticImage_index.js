let net;
const webcamElement = document.getElementById('webcam');

async function app() {
  console.log('Loading model..');

  // Load the model.
  // net = await mobilenet.load();
  const net = await tf.automl.loadImageClassification('model.json');
  console.log('Successfully loaded model');
  
  // Create an object from Tensorflow.js data API which could capture image 
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);
  while (true) {
    const img = await webcam.capture();
    const result = await net.classify(img);

    console.log(result);
    document.getElementById('console').innerText = `
      prediction: ${result[0].label}\n
      probability: ${result[0].prob}
    `;
    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

// function readURL(input) {
// 	if (input.files && input.files[0]) {
// 			var reader = new FileReader();

// 			reader.onload = function (e) {
// 				img = document.getElementById('imageUploaded');
// 				img.setAttribute('src', e.target.result);
// 				img.width(250);
// 				img.height(200);
// 			};

// 			reader.readAsDataURL(input.files[0]);
// 	}
// }

app();