using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RealTimeFaceRecognition.Detection
{
    public class FaceDetectionDNNService : IDisposable
    {
        private readonly Net _faceNet;
        private readonly float _confThreshold;

        public FaceDetectionDNNService(string modelConfiguration, string modelWeights, float confThreshold = 0.5f)
        {
            // Cargar la red preentrenada
            _faceNet = DnnInvoke.ReadNetFromCaffe(modelConfiguration, modelWeights);
            _confThreshold = confThreshold;
        }

        public Rectangle[] DetectFaces(Mat frame)
        {
            // Preparar la imagen para la red
            Mat blob = DnnInvoke.BlobFromImage(frame, 1.0, new Size(300, 300), new MCvScalar(104, 177, 123), false, false);

            // Establecer el blob como entrada de la red
            _faceNet.SetInput(blob);

            // Ejecutar la detección
            Mat detections = _faceNet.Forward();

            // Parsear las detecciones
            var faces = new System.Collections.Generic.List<Rectangle>();

            for (int i = 0; i < detections.SizeOfDimension[2]; i++)
            {
                float confidence = detections.GetData().GetValue(0, 0, i, 2) != null ? (float)detections.GetData().GetValue(0, 0, i, 2) : 0f;

                if (confidence > _confThreshold)
                {
                    int x1 = (int)((float)detections.GetData().GetValue(0, 0, i, 3) * frame.Cols);
                    int y1 = (int)((float)detections.GetData().GetValue(0, 0, i, 4) * frame.Rows);
                    int x2 = (int)((float)detections.GetData().GetValue(0, 0, i, 5) * frame.Cols);
                    int y2 = (int)((float)detections.GetData().GetValue(0, 0, i, 6) * frame.Rows);

                    // Asegurarse de que las coordenadas estén dentro de la imagen
                    x1 = Math.Max(0, x1);
                    y1 = Math.Max(0, y1);
                    x2 = Math.Min(frame.Cols - 1, x2);
                    y2 = Math.Min(frame.Rows - 1, y2);

                    Rectangle faceRect = new Rectangle(x1, y1, x2 - x1, y2 - y1);
                    faces.Add(faceRect);
                }
            }

            return faces.ToArray();
        }

        public void Dispose()
        {
            _faceNet?.Dispose();
        }
    }
}
