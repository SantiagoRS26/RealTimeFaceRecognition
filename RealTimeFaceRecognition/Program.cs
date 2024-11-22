using BLL.Interfaces;
using BLL.Services;
using DAL.Interfaces;
using DAL.Servicios;
using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Drawing;

namespace RealTimeFaceRecognition
{
    internal class Program
    {
        private static IFaceDetectionService _faceDetectionService;

        static void Main(string[] args)
        {
            // Rutas a los archivos del modelo DNN
            string modelConfiguration = "deploy.prototxt";
            string modelWeights = "res10_300x300_ssd_iter_140000.caffemodel";

            // Configurar los servicios
            ConfigureServices(modelConfiguration, modelWeights, 0.8f);

            using (IVideoCaptureService captureService = new VideoCaptureService(0))
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

            _faceDetectionService.Dispose();
        }

        private static void ConfigureServices(string modelConfiguration, string modelWeights, float confThreshold)
        {
            // Crear el cargador de modelo (DAL)
            IFaceModelLoader modelLoader = new FaceModelLoader();

            // Crear el servicio de detección (BLL) utilizando el modelo cargador
            _faceDetectionService = new FaceDetectionDNNService(modelLoader, modelConfiguration, modelWeights, confThreshold);
        }

        private static void OnFrameCaptured(object sender, Mat frame)
        {
            // Detectar rostros en el frame usando DNN
            Rectangle[] faces = _faceDetectionService.DetectFaces(frame);

            // Actualizar el contador de personas
            int personCount = faces.Length;

            // Dibujar rectángulos alrededor de los rostros detectados
            foreach (var face in faces)
            {
                CvInvoke.Rectangle(frame, face, new MCvScalar(0, 255, 0), 2);
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
