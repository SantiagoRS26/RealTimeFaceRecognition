using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV.Util;
using static Emgu.CV.Face.FaceRecognizer;

namespace RealTimeFaceRecognition.Recognition
{
    public class FaceRecognitionService : IDisposable
    {
        private LBPHFaceRecognizer _faceRecognizer;
        private Dictionary<int, string> _labelNameMap;
        private bool _isTrained = false;

        public FaceRecognitionService()
        {
            _faceRecognizer = new LBPHFaceRecognizer(1, 8, 8, 8, 123);
            _labelNameMap = new Dictionary<int, string>();
        }

        private Image<Gray, byte> PreprocessImage(string imagePath)
        {
            // Cargar la imagen en escala de grises
            var img = new Image<Gray, byte>(imagePath);

            // Redimensionar a 200x200 píxeles
            img = img.Resize(200, 200, Emgu.CV.CvEnum.Inter.Linear);

            // Aplicar ecualización del histograma
            img._EqualizeHist();

            // Aplicar desenfoque Gaussiano para eliminar ruido
            img = img.SmoothGaussian(3);

            return img;
        }

        private Image<Gray, byte> PreprocessImage(Image<Gray, byte> img)
        {
            // Redimensionar a 200x200 píxeles
            img = img.Resize(200, 200, Emgu.CV.CvEnum.Inter.Linear);

            // Aplicar ecualización del histograma
            img._EqualizeHist();

            // Aplicar desenfoque Gaussiano para eliminar ruido
            img = img.SmoothGaussian(3);

            return img;
        }

        private List<Image<Gray, byte>> AugmentImage(Image<Gray, byte> img)
        {
            List<Image<Gray, byte>> augmentedImages = new List<Image<Gray, byte>>();

            // Rotación de -15 y +15 grados
            for (int angle = -15; angle <= 15; angle += 30)
            {
                if (angle == 0) continue;
                var rotated = img.Rotate(angle, new Gray(0));
                augmentedImages.Add(rotated);
            }

            // Reflejo horizontal
            var flipped = img.Flip(Emgu.CV.CvEnum.FlipType.Horizontal);
            augmentedImages.Add(flipped);

            // Cambios de brillo
            var bright = img + new Gray(50); // Aumenta la intensidad de los píxeles
            augmentedImages.Add(bright);

            var dark = img - new Gray(50); // Disminuye la intensidad de los píxeles
            augmentedImages.Add(dark);

            // Cambios de contraste
            var highContrast = img * 1.5; // Aumenta el contraste
            augmentedImages.Add(highContrast);

            var lowContrast = img * 0.7; // Disminuye el contraste
            augmentedImages.Add(lowContrast);

            return augmentedImages;
        }




        public void TrainRecognizer(string knownFacesPath)
        {
            if (!Directory.Exists(knownFacesPath))
            {
                throw new DirectoryNotFoundException($"La carpeta {knownFacesPath} no existe.");
            }

            List<Image<Gray, byte>> faceImages = new List<Image<Gray, byte>>();
            List<int> labels = new List<int>();
            int currentLabel = 0;

            foreach (var personDir in Directory.GetDirectories(knownFacesPath))
            {
                string personName = Path.GetFileName(personDir);
                _labelNameMap[currentLabel] = personName;

                foreach (var imagePath in Directory.GetFiles(personDir, "*.jpg"))
                {
                    try
                    {
                        // Preprocesar la imagen antes de agregarla
                        var img = PreprocessImage(imagePath);
                        faceImages.Add(img);
                        labels.Add(currentLabel);

                        // Generar imágenes aumentadas
                        var augmentedImages = AugmentImage(img);
                        foreach (var augImg in augmentedImages)
                        {
                            faceImages.Add(augImg);
                            labels.Add(currentLabel);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error al cargar la imagen {imagePath}: {ex.Message}");
                    }
                }

                currentLabel++;
            }

            if (faceImages.Count == 0)
            {
                throw new Exception("No se encontraron imágenes de referencia para entrenar el reconocedor.");
            }

            // Convertir listas a VectorOfMat y VectorOfInt
            using (var trainingImages = new VectorOfMat())
            using (var trainingLabels = new VectorOfInt(labels.ToArray()))
            {
                foreach (var img in faceImages)
                {
                    trainingImages.Push(img.Mat);
                }

                // Entrenar el reconocedor
                _faceRecognizer.Train(trainingImages, trainingLabels);
            }

            _isTrained = true;

            Console.WriteLine($"Reconocedor entrenado con {faceImages.Count} imágenes de {currentLabel} personas.");
        }



        public string RecognizeFace(Image<Gray, byte> faceImage, out int label, out double confidence)
        {
            if (!_isTrained)
            {
                throw new InvalidOperationException("El reconocedor no ha sido entrenado.");
            }

            // Preprocesar la imagen de la cara antes de reconocerla
            faceImage = PreprocessImage(faceImage);

            // Obtener el resultado de la predicción
            PredictionResult result = _faceRecognizer.Predict(faceImage.Mat);

            label = result.Label;
            confidence = result.Distance;

            // Define un umbral de confianza para considerar una predicción como válida
            double confidenceThreshold = 100.0; // Ajusta este valor según tus necesidades

            if (confidence < confidenceThreshold && _labelNameMap.ContainsKey(label))
            {
                return _labelNameMap[label];
            }
            else
            {
                return "Desconocido";
            }
        }


        public void Dispose()
        {
            _faceRecognizer?.Dispose();
        }
    }
}
