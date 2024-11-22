using BLL.Interfaces;
using Emgu.CV.Structure;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Windows.Threading;
using System.Drawing;
using Models.Domain;
using BLL.Services;
using Microsoft.Extensions.DependencyInjection;

namespace FaceDetection.ViewModels
{
    public class MainViewModel : INotifyPropertyChanged, IDisposable
    {
        private readonly IVideoCaptureService _videoCaptureService;
        private readonly IFaceDetectionService _faceDetectionService;
        private readonly IServiceScopeFactory _scopeFactory;

        private ImageSource _currentFrame;
        private readonly Dispatcher _dispatcher;

        public event PropertyChangedEventHandler PropertyChanged;

        private bool _isIntervalActive = false;
        private DateTime _intervalStartTime;
        private Interval _currentInterval;
        private List<Detection> _currentDetections;
        private int _maxPersonsInInterval = 0;

        private DispatcherTimer _intervalTimer;

        private VideoWriter _videoWriter;

        // Agregamos el campo _currentVideo
        private Video _currentVideo;

        public ImageSource CurrentFrame
        {
            get => _currentFrame;
            private set
            {
                _currentFrame = value;
                OnPropertyChanged(nameof(CurrentFrame));
            }
        }

        public MainViewModel(
            IVideoCaptureService videoCaptureService,
            IFaceDetectionService faceDetectionService,
            IServiceScopeFactory scopeFactory)
        {
            _videoCaptureService = videoCaptureService;
            _faceDetectionService = faceDetectionService;
            _scopeFactory = scopeFactory;
            _dispatcher = Dispatcher.CurrentDispatcher;

            _currentDetections = new List<Detection>();

            _intervalTimer = new DispatcherTimer();
            _intervalTimer.Interval = TimeSpan.FromSeconds(60);
            _intervalTimer.Tick += OnIntervalTimerTick;

            _videoCaptureService.FrameCaptured += OnFrameCaptured;
            _videoCaptureService.Start();
        }

        private async void OnFrameCaptured(object sender, Mat frame)
        {
            try
            {
                // Realizar la detección de rostros
                Rectangle[] faces = _faceDetectionService.DetectFaces(frame);

                // Dibujar rectángulos alrededor de los rostros
                foreach (var face in faces)
                {
                    CvInvoke.Rectangle(frame, face, new Emgu.CV.Structure.MCvScalar(0, 255, 0), 2);
                }

                // Actualizar la imagen en la interfaz de usuario
                _dispatcher.Invoke(() =>
                {
                    CurrentFrame = ConvertMatToImageSource(frame);
                });

                // Si se detecta al menos una persona y no hay un intervalo activo, iniciamos uno
                if (faces.Length > 0 && !_isIntervalActive)
                {
                    await StartIntervalAsync();
                }

                // Si hay un intervalo activo, almacenamos las detecciones y escribimos el frame en el video
                if (_isIntervalActive && _videoWriter != null)
                {
                    _videoWriter.Write(frame);
                    await SaveDetectionsAsync(faces);
                }
            }
            catch (Exception ex)
            {
                // Manejo de errores
                using (var scope = _scopeFactory.CreateScope())
                {
                    var logService = scope.ServiceProvider.GetRequiredService<ILogService>();
                    await logService.AddLogAsync(new Log
                    {
                        LogTimestamp = DateTime.UtcNow,
                        Level = "Error",
                        Event = $"Error en OnFrameCaptured: {ex.Message}",
                        VideoId = _currentVideo?.VideoId,
                        IntervalId = _currentInterval?.IntervalId
                    });
                }
            }
        }

        private async Task SaveDetectionsAsync(Rectangle[] faces)
        {
            // Actualizar el máximo de personas detectadas en el intervalo
            if (faces.Length > _maxPersonsInInterval)
            {
                _maxPersonsInInterval = faces.Length;
            }

            using (var scope = _scopeFactory.CreateScope())
            {
                var detectionService = scope.ServiceProvider.GetRequiredService<IDetectionService>();

                // Guardar cada detección
                foreach (var face in faces)
                {
                    var detection = new Detection
                    {
                        IntervalId = _currentInterval.IntervalId,
                        Timestamp = DateTime.UtcNow,
                        PositionX = face.X,
                        PositionY = face.Y,
                        Width = face.Width,
                        Height = face.Height,
                        Confidence = 0.5f // Reemplaza con la confianza real si está disponible
                    };

                    await detectionService.AddDetectionAsync(detection);

                    // Añadir a la lista actual de detecciones
                    _currentDetections.Add(detection);
                }
            }
        }

        private void StartRecordingInterval()
        {
            // Define el nombre del archivo y la ruta
            string videoFileName = $"Interval_{DateTime.UtcNow:yyyyMMdd_HHmmss}.avi";
            string videoFilePath = Path.Combine("Videos", videoFileName);

            // Asegúrate de que el directorio existe
            Directory.CreateDirectory("Videos");

            // Inicializa el VideoWriter
            int fourcc = VideoWriter.Fourcc('M', 'J', 'P', 'G');
            double fps = _videoCaptureService.GetFramesPerSecond();
            int width = _videoCaptureService.GetFrameWidth();
            int height = _videoCaptureService.GetFrameHeight();

            _videoWriter = new VideoWriter(videoFilePath, fourcc, fps, new System.Drawing.Size(width, height), true);

            // Crear y asignar un nuevo Video
            _currentVideo = new Video
            {
                FilePath = videoFilePath,
                StartTime = _intervalStartTime,
                // EndTime se actualizará al detener la grabación
            };
        }

        private async Task StopRecordingIntervalAsync()
        {
            if (_videoWriter != null)
            {
                _videoWriter.Dispose();
                _videoWriter = null;

                // Actualizar el EndTime y DurationInSeconds del video
                _currentVideo.EndTime = DateTime.UtcNow;
                _currentVideo.DurationInSeconds = (int)(_currentVideo.EndTime - _currentVideo.StartTime).TotalSeconds;

                using (var scope = _scopeFactory.CreateScope())
                {
                    var videoService = scope.ServiceProvider.GetRequiredService<IVideoService>();
                    var intervalService = scope.ServiceProvider.GetRequiredService<IIntervalService>();

                    // Guardar el Video en la base de datos
                    await videoService.AddVideoAsync(_currentVideo);

                    // Ahora que el Video tiene un VideoId, asignamos el VideoId al Intervalo
                    _currentInterval.VideoId = _currentVideo.VideoId;

                    // Actualizar el Intervalo en la base de datos
                    await intervalService.UpdateIntervalAsync(_currentInterval);
                }
            }
        }

        private async void OnIntervalTimerTick(object sender, EventArgs e)
        {
            _intervalTimer.Stop();
            _isIntervalActive = false;

            // Actualizar el EndTime del intervalo
            _currentInterval.EndTime = DateTime.UtcNow;

            using (var scope = _scopeFactory.CreateScope())
            {
                var intervalService = scope.ServiceProvider.GetRequiredService<IIntervalService>();
                await intervalService.UpdateIntervalAsync(_currentInterval);
            }

            // Detener la grabación del video del intervalo
            await StopRecordingIntervalAsync();

            // Calcular y actualizar estadísticas del intervalo
            await UpdateIntervalStatisticsAsync();

            // Reiniciar variables
            _currentInterval = null;
        }

        private async Task UpdateIntervalStatisticsAsync()
        {
            var detections = _currentDetections;

            // Calcular el promedio de personas detectadas por frame
            float averagePersons = detections.Count > 0 ? (float)detections.Count / 60f : 0f;

            // Actualizar los datos del intervalo
            _currentInterval.MaxPersons = _maxPersonsInInterval;
            _currentInterval.AveragePersons = averagePersons;

            using (var scope = _scopeFactory.CreateScope())
            {
                var intervalService = scope.ServiceProvider.GetRequiredService<IIntervalService>();
                await intervalService.UpdateIntervalAsync(_currentInterval);
            }
        }

        private async Task StartIntervalAsync()
        {
            _isIntervalActive = true;
            _intervalStartTime = DateTime.UtcNow;
            _currentDetections.Clear();
            _maxPersonsInInterval = 0;

            // Crear un nuevo intervalo sin VideoId
            _currentInterval = new Interval
            {
                StartTime = _intervalStartTime,
                EndTime = _intervalStartTime.AddSeconds(60),
                Notes = "Intervalo iniciado automáticamente al detectar una persona."
            };

            using (var scope = _scopeFactory.CreateScope())
            {
                var intervalService = scope.ServiceProvider.GetRequiredService<IIntervalService>();
                await intervalService.AddIntervalAsync(_currentInterval);
            }

            // Iniciar el timer de 60 segundos
            _intervalTimer.Start();

            // Iniciar la grabación del video del intervalo
            StartRecordingInterval();
        }

        private ImageSource ConvertMatToImageSource(Mat mat)
        {
            BitmapImage bitmap = new BitmapImage();
            using (MemoryStream ms = new MemoryStream())
            {
                mat.ToImage<Bgr, byte>().ToBitmap().Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
                ms.Position = 0;

                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.StreamSource = ms;
                bitmap.EndInit();
                bitmap.Freeze();
            }
            return bitmap;
        }

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public void Dispose()
        {
            _videoCaptureService?.Dispose();
            _faceDetectionService?.Dispose();
        }
    }
}
