using BLL.Interfaces;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Services
{
    public class VideoCaptureService : IVideoCaptureService
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

        public double GetFramesPerSecond()
        {
            // Obtener el FPS de la captura
            double fps = _capture.Get(Emgu.CV.CvEnum.CapProp.Fps);
            if (fps == 0 || double.IsNaN(fps))
            {
                fps = 30; // Valor predeterminado si no se puede obtener el FPS
            }
            return fps;
        }

        public int GetFrameWidth()
        {
            return (int)_capture.Get(Emgu.CV.CvEnum.CapProp.FrameWidth);
        }

        public int GetFrameHeight()
        {
            return (int)_capture.Get(Emgu.CV.CvEnum.CapProp.FrameHeight);
        }

    }
}
