using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RealTimeFaceRecognition.Capture
{
    public class VideoCaptureService : IDisposable
    {
        private VideoCapture _capture;
        private int _cameraIndex;
        private bool _isRunning;

        // Evento para pasar el frame capturado
        public event EventHandler<Mat> FrameCaptured;

        public VideoCaptureService(int cameraIndex = 0)
        {
            _cameraIndex = cameraIndex;
            _capture = new VideoCapture(_cameraIndex);
            _isRunning = false;
        }

        public void Start()
        {
            if (!_isRunning)
            {
                _isRunning = true;
                _capture.ImageGrabbed += ProcessFrame;
                _capture.Start();
            }
        }

        public void Stop()
        {
            if (_isRunning)
            {
                _isRunning = false;
                _capture.ImageGrabbed -= ProcessFrame;
                _capture.Stop();
            }
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            if (!_isRunning) return;

            Mat frame = new Mat();
            _capture.Retrieve(frame);

            // Disparar el evento con el frame capturado
            FrameCaptured?.Invoke(this, frame);
        }

        public void Dispose()
        {
            Stop();
            _capture.Dispose();
        }
    }
}
