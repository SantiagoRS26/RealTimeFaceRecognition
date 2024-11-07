using Emgu.CV;
using Emgu.CV.Structure;
using RealTimeFaceRecognition.Capture;
using RealTimeFaceRecognition.Detection;
using RealTimeFaceRecognition.Recognition;
using System;
using System.Drawing;

namespace RealTimeFaceRecognition
{
    internal class Program
    {
        static FaceDetectionDNNService _faceDetectionDNNService;
        static FaceRecognitionService _faceRecognitionService;

        static int _totalPersonCount = 0;

        static void Main(string[] args)
        {
            // Rutas a los archivos del modelo DNN
            string modelConfiguration = "deploy.prototxt";
            string modelWeights = "res10_300x300_ssd_iter_140000.caffemodel";

            // Inicializar el servicio de detección de rostros basado en DNN
            _faceDetectionDNNService = new FaceDetectionDNNService(modelConfiguration, modelWeights, confThreshold: 0.8f);

            // Inicializar el servicio de reconocimiento facial
            _faceRecognitionService = new FaceRecognitionService();
            string knownFacesPath = "C:\\Users\\santi\\source\\repos\\RealTimeFaceRecognition\\RealTimeFaceRecognition\\KnownFaces\\"; // Asegúrate de que esta carpeta exista y esté correctamente estructurada
            try
            {
                _faceRecognitionService.TrainRecognizer(knownFacesPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al entrenar el reconocedor: {ex.Message}");
                return; // Salir si no se puede entrenar
            }

            using (VideoCaptureService captureService = new VideoCaptureService(0))
            {
                captureService.FrameCaptured += OnFrameCaptured;
                captureService.Start();

                Console.WriteLine("Presiona 'Esc' para salir...");
                while (Console.ReadKey(true).Key != ConsoleKey.Escape)
                {
                    // Espera a que el usuario presione 'Esc' para salir
                }

                captureService.Stop();
            }

            _faceDetectionDNNService.Dispose();
            _faceRecognitionService.Dispose();
        }

        private static void OnFrameCaptured(object sender, Mat frame)
        {
            // Detectar rostros en el frame usando DNN
            Rectangle[] faces = _faceDetectionDNNService.DetectFaces(frame);

            // Actualizar el contador de personas
            int personCount = faces.Length;

            // Dibujar rectángulos alrededor de los rostros detectados y reconocer
            foreach (var face in faces)
            {
                // Extraer la región facial
                var faceRegion = new Rectangle(face.X, face.Y, face.Width, face.Height);
                Mat faceMat = new Mat(frame, faceRegion);

                // Preprocesar la imagen de la cara para el reconocimiento
                Image<Gray, byte> grayFace = faceMat.ToImage<Gray, byte>();
                grayFace = grayFace.Resize(200, 200, Emgu.CV.CvEnum.Inter.Linear); // Asegúrate de que el tamaño coincida con el entrenamiento

                // Reconocer la cara
                string label = "Desconocido";
                int predictedLabel;
                double confidence = 0.0;
                try
                {
                    label = _faceRecognitionService.RecognizeFace(grayFace, out predictedLabel, out confidence);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error al reconocer la cara: {ex.Message}");
                }

                // Dibujar el rectángulo alrededor de la cara
                CvInvoke.Rectangle(frame, faceRegion, new MCvScalar(0, 255, 0), 2);

                // Escribir el nombre de la persona
                string text = $"{label} ({confidence:F2})";
                Point textPoint = new Point(faceRegion.X, faceRegion.Y - 10 > 10 ? faceRegion.Y - 10 : faceRegion.Y + faceRegion.Height + 20);
                CvInvoke.PutText(frame, text, textPoint, Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.8, new MCvScalar(0, 255, 0), 2);
            }

            // Mostrar el conteo en la imagen
            string countText = $"Personas Detectadas: {personCount}";
            CvInvoke.PutText(frame, countText, new Point(10, 30), Emgu.CV.CvEnum.FontFace.HersheySimplex, 1.0, new MCvScalar(0, 255, 0), 2);

            // Mostrar el frame con los rostros detectados y el conteo
            CvInvoke.Imshow("Video en Tiempo Real - DNN Face Detection", frame);
            CvInvoke.WaitKey(1);
        }
    }
}
