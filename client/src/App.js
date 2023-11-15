// App.js
import React, { useState, useRef } from 'react';
import axios from 'axios';
import ProgressBar from './Loader';
import './App.css'
function App() {
  const [pred, setPred] = useState("")
  const videoRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState(null);
const [load, setLoad] = useState(0)
  const startRecording = () => {
    var video =document.getElementById('vid')
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
          video.srcObject = stream;
        })
        .catch(function (err0r) {
          console.log("Something went wrong!");
        });
    }
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        const mediaRecorder = new MediaRecorder(stream);
        const chunks = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunks.push(event.data);
          }
        };

        mediaRecorder.onstop = () => {
          const videoBlob = new Blob(chunks, { type: 'video/mp4' });
          setVideoBlob(videoBlob);
        };

        mediaRecorder.start();
        setIsRecording(true);

        setTimeout(() => {
          mediaRecorder.stop();
          stream.getTracks().forEach((track) => track.stop());
        }, 5000); // Record for 5 seconds (adjust as needed)
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  const sendVideoToBackend = () => {
    const formData = new FormData();
    formData.append('video', videoBlob);
    setLoad(50)
    axios.post('http://127.0.0.1:5000/upload_video', formData,{
      headers:{
        'Content-Type': 'multipart/form-data',
      }
    })
      .then((response) => {
        // Handle the response from the Flask backend
        console.log(response.data)
        setPred(response.data['pred'])
        setLoad(100)
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  return (
    <div className="App">

      <h1 style={{fontFamily:"sans-serif"}}>
        Hand Sign to text generator
      </h1>
      <video className='h-55' id='vid'  controls autoPlay={true}/>
      <div className='my-3'>
      <button onClick={startRecording} disabled={isRecording}>
        Start Recording
      </button>
      <button onClick={sendVideoToBackend} disabled={!videoBlob}>
        Predict
      </button>
      <button onClick={()=>{window.location.reload()}} disabled={!videoBlob}>
        New recording
      </button>
      </div>
      <div className='w-50 my-1' >

    <ProgressBar progressPercentage={load}/>
      </div>
      <div style={{display:'flex',gap:"1.2rem",alignItems:"center",justifyContent:"space-between",width:"25%"}}>
        <h2 style={{fontFamily:"sans-serif"}}>Prediction</h2>
        <div style={{fontFamily:"sans-serif",fontSize:"1.1rem",fontWeight:'bold',color:"green"}}>
          {pred===""?"No result":pred}
        </div>
      </div>
    </div>
  );
}

export default App;
