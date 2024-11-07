using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RealTimeFaceRecognition.Detection
{
    public class FaceDetectionService : IDisposable
    {
        private CascadeClassifier _faceCascade;

        public FaceDetectionService(string cascadeFilePath)
        {
            _faceCascade = new CascadeClassifier(cascadeFilePath);
        }

        public Rectangle[] DetectFaces(Mat frame)
        {
            using (var grayFrame = new UMat())
            {
                CvInvoke.CvtColor(frame, grayFrame, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
                CvInvoke.EqualizeHist(grayFrame, grayFrame);

                Rectangle[] faces = _faceCascade.DetectMultiScale(
                    grayFrame,
                    scaleFactor: 1.1,
                    minNeighbors: 5,
                    minSize: new Size(30, 30),
                    maxSize: Size.Empty);

                return faces;
            }
        }
        public void Dispose()
        {
            _faceCascade.Dispose();
        }
    }
}
