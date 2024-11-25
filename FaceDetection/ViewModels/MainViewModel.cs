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
using Amazon.S3;

namespace FaceDetection.ViewModels
{
    public class MainViewModel : INotifyPropertyChanged, IDisposable
    {
        private readonly IVideoCaptureService _videoCaptureService;
        private readonly IFaceDetectionService _faceDetectionService;
        private readonly IServiceScopeFactory _scopeFactory;
        private readonly IS3Service _s3Service;

        private ImageSource _currentFrame;
        private readonly Dispatcher _dispatcher;

        public event PropertyChangedEventHandler PropertyChanged;

        private bool _isIntervalActive = false;
        private DateTime _intervalStartTime;
        private int _maxPersonsInInterval = 0;
        private int _totalFacesDetected = 0;

        private DispatcherTimer _intervalTimer;

        private VideoWriter _videoWriter;

        private Video _currentVideo;

        private readonly object _videoWriterLock = new object();

        // Flags y variables para gestionar logs de entrada/salida
        private bool _hasLoggedPersonDetected = false;
        private int _previousFaceCount = 0;
        private DateTime _lastPersonLogTime = DateTime.MinValue;
        private readonly TimeSpan _personLogCooldown = TimeSpan.FromSeconds(3); // Período de enfriamiento de 3 segundos

        public ImageSource CurrentFrame
        {
            get => _currentFrame;
            private set
            {
                _currentFrame = value;
                OnPropertyChanged(nameof(CurrentFrame));
            }
        }

        private int _currentPersonCount;
        public int CurrentPersonCount
        {
            get => _currentPersonCount;
            private set
            {
                if (_currentPersonCount != value)
                {
                    _currentPersonCount = value;
                    OnPropertyChanged(nameof(CurrentPersonCount));
                }
            }
        }


        public MainViewModel(
            IVideoCaptureService videoCaptureService,
            IFaceDetectionService faceDetectionService,
            IServiceScopeFactory scopeFactory,
            IS3Service s3Service)
        {
            _videoCaptureService = videoCaptureService;
            _faceDetectionService = faceDetectionService;
            _scopeFactory = scopeFactory;
            _dispatcher = Dispatcher.CurrentDispatcher;

            _intervalTimer = new DispatcherTimer();
            _intervalTimer.Interval = TimeSpan.FromSeconds(20);
            _intervalTimer.Tick += OnIntervalTimerTick;

            _videoCaptureService.FrameCaptured += OnFrameCaptured;
            _videoCaptureService.Start();
            _s3Service = s3Service;
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

                // Actualizar el contador de personas detectadas
                _dispatcher.Invoke(() =>
                {
                    CurrentPersonCount = faces.Length;
                });

                // Rastrear cambios en el número de caras para detectar entradas y salidas
                int currentFaceCount = faces.Length;
                DateTime currentTime = DateTime.UtcNow;

                // Detectar entrada de personas
                if (currentFaceCount > _previousFaceCount &&
                    (currentTime - _lastPersonLogTime) > _personLogCooldown)
                {
                    _lastPersonLogTime = currentTime;
                    await CreateLogAsync("Persona ingresó");
                }
                // Detectar salida de personas
                else if (currentFaceCount < _previousFaceCount &&
                         (currentTime - _lastPersonLogTime) > _personLogCooldown)
                {
                    _lastPersonLogTime = currentTime;
                    await CreateLogAsync("Persona salió");
                }

                // Actualizar el conteo previo
                _previousFaceCount = currentFaceCount;

                // Si se detecta al menos una persona y no hay un intervalo activo, iniciar uno
                if (faces.Length > 0 && !_isIntervalActive)
                {
                    await StartIntervalAsync();
                }

                // Crear log de "Persona detectada" una vez por intervalo
                if (_isIntervalActive && faces.Length > 0 && !_hasLoggedPersonDetected)
                {
                    _hasLoggedPersonDetected = true;

                    // Crear log de detección de persona
                    await CreateLogAsync("Persona detectada");
                }

                // Si hay un intervalo activo, escribir el frame en el video y actualizar las estadísticas
                if (_isIntervalActive && _videoWriter != null)
                {
                    if (frame == null || frame.IsEmpty)
                    {
                        Console.WriteLine("Frame está vacío o nulo, no se escribe en el video.");
                        return;
                    }

                    try
                    {
                        lock (_videoWriterLock)
                        {
                            _videoWriter.Write(frame);
                        }

                        // Actualizar estadísticas
                        if (faces.Length > _maxPersonsInInterval)
                        {
                            _maxPersonsInInterval = faces.Length;
                        }
                        _totalFacesDetected += faces.Length;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error al escribir el frame en el video: {ex.Message}");

                        // Crear un log de error al escribir el frame
                        await CreateLogAsync($"Error al escribir el frame en el video: {ex.Message}");
                    }
                }

            }
            catch (Exception ex)
            {
                // Manejo de errores
                await CreateLogAsync($"Error en OnFrameCaptured: {ex.Message}");
            }
        }


        private async Task CreateLogAsync(string eventDescription)
        {
            // Validar que exista un video activo antes de crear el log
            if (_currentVideo == null)
            {
                Console.WriteLine("No hay un Video activo. No se puede crear el log.");
                return;
            }

            // Validar que el VideoId sea válido
            if (_currentVideo.VideoId == 0)
            {
                Console.WriteLine("El VideoId no es válido. No se puede crear el log.");
                return;
            }

            // Crear el log
            using (var scope = _scopeFactory.CreateScope())
            {
                var logService = scope.ServiceProvider.GetRequiredService<ILogService>();
                await logService.AddLogAsync(new Log
                {
                    Timestamp = DateTime.UtcNow,
                    Event = eventDescription,
                    VideoId = _currentVideo.VideoId
                });
            }

            Console.WriteLine($"Log creado: {eventDescription} para VideoId: {_currentVideo.VideoId}");
        }

        private void StartRecordingInterval()
        {
            // El directorio ya se ha creado en StartIntervalAsync

            int width = _videoCaptureService.GetFrameWidth();
            int height = _videoCaptureService.GetFrameHeight();
            double fps = _videoCaptureService.GetFramesPerSecond();

            // Validar tamaño y FPS
            Console.WriteLine($"VideoWriter Configuración: Width={width}, Height={height}, FPS={fps}");

            // Inicializar VideoWriter con el tamaño y FPS correctos
            // Utilizar MJPG para .avi
            int fourcc = VideoWriter.Fourcc('M', 'J', 'P', 'G');
            string tempFilePath = Path.Combine("Videos", $"Temp_{DateTime.UtcNow:yyyyMMdd_HHmmss}.avi");
            _videoWriter = new VideoWriter(tempFilePath, fourcc, fps, new Size(width, height), true);

            // Asignar el FilePath temporal al Video actual
            _currentVideo = new Video
            {
                StartTime = _intervalStartTime,
                FilePath = tempFilePath,
                // Otros campos se actualizarán al detener la grabación
            };
        }

        private async Task StopRecordingIntervalAsync()
        {
            if (_videoWriter != null)
            {
                lock (_videoWriterLock)
                {
                    _videoWriter.Dispose();
                    _videoWriter = null;
                }
            }

            if (_currentVideo != null)
            {
                // Actualizar metadatos del video
                _currentVideo.EndTime = DateTime.UtcNow;
                _currentVideo.DurationInSeconds = (int)(_currentVideo.EndTime - _currentVideo.StartTime).TotalSeconds;
                _currentVideo.MaxPersons = _maxPersonsInInterval;
                _currentVideo.AveragePersons = _currentVideo.DurationInSeconds > 0
                    ? (float)_totalFacesDetected / _currentVideo.DurationInSeconds
                    : 0;

                using (var scope = _scopeFactory.CreateScope())
                {
                    var videoService = scope.ServiceProvider.GetRequiredService<IVideoService>();
                    await videoService.UpdateVideoAsync(_currentVideo);

                    // Crear log de fin de grabación
                    await CreateLogAsync("Fin de grabación");
                }

                // Convertir AVI a MP4 usando FFmpeg embebido
                string aviFilePath = _currentVideo.FilePath;
                string mp4FilePath = Path.Combine("Videos", $"Interval_{DateTime.UtcNow:yyyyMMdd_HHmmss}.mp4");

                try
                {
                    // Ruta al ejecutable de FFmpeg embebido
                    string ffmpegPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "FFmpeg", "ffmpeg.exe");

                    // Validar si FFmpeg existe en la ruta configurada
                    if (!File.Exists(ffmpegPath))
                    {
                        throw new FileNotFoundException("No se encontró el ejecutable de FFmpeg en la ruta especificada.", ffmpegPath);
                    }

                    var ffmpegProcess = new System.Diagnostics.Process
                    {
                        StartInfo = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = ffmpegPath,
                            Arguments = $"-i \"{aviFilePath}\" -vcodec libx264 -acodec aac \"{mp4FilePath}\" -y",
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            UseShellExecute = false,
                            CreateNoWindow = true,
                        }
                    };

                    ffmpegProcess.Start();
                    string output = await ffmpegProcess.StandardError.ReadToEndAsync();
                    ffmpegProcess.WaitForExit();

                    if (ffmpegProcess.ExitCode != 0)
                    {
                        throw new Exception($"FFmpeg error: {output}");
                    }

                    // Subir el archivo MP4 a S3
                    if (File.Exists(mp4FilePath))
                    {
                        string keyName = Path.GetFileName(mp4FilePath);
                        string s3Url = await _s3Service.UploadFileAsync(mp4FilePath, keyName);

                        // Guardar la URL de S3 en el objeto Video o en la base de datos
                        _currentVideo.S3Url = s3Url;

                        using (var scope = _scopeFactory.CreateScope())
                        {
                            var videoService = scope.ServiceProvider.GetRequiredService<IVideoService>();
                            await videoService.UpdateVideoAsync(_currentVideo);
                        }

                        Console.WriteLine($"Video subido a S3: {s3Url}");

                        // Opcional: Eliminar los archivos locales después de la subida
                        try
                        {
                            File.Delete(aviFilePath);
                            File.Delete(mp4FilePath);
                            Console.WriteLine("Archivos locales eliminados después de subir a S3.");
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error al eliminar los archivos locales: {ex.Message}");
                            await CreateLogAsync($"Error al eliminar los archivos locales: {ex.Message}");
                        }
                    }
                    else
                    {
                        throw new FileNotFoundException("El archivo de video MP4 no existe.", mp4FilePath);
                    }
                }
                catch (FileNotFoundException ex)
                {
                    Console.WriteLine($"Error: {ex.Message}");
                    await CreateLogAsync($"Error: {ex.Message}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error al procesar el video: {ex.Message}");
                    await CreateLogAsync($"Error al procesar el video: {ex.Message}");
                }

            }

            // Reiniciar variables
            _currentVideo = null;
            _hasLoggedPersonDetected = false; // Reset para el próximo intervalo
        }


        private async void OnIntervalTimerTick(object sender, EventArgs e)
        {
            _intervalTimer.Stop();
            _isIntervalActive = false;

            // Detener la grabación del video del intervalo
            await StopRecordingIntervalAsync();

            // Reiniciar variables
            _currentVideo = null;
            _hasLoggedPersonDetected = false;
            _previousFaceCount = 0;
        }

        private async Task StartIntervalAsync()
        {
            _isIntervalActive = true;
            _hasLoggedPersonDetected = true; // Prevenir la creación de "Persona detectada" en este frame
            _intervalStartTime = DateTime.UtcNow;
            _maxPersonsInInterval = 0;
            _totalFacesDetected = 0;

            // Iniciar la grabación del video del intervalo
            StartRecordingInterval();

            // Guardar el Video en la base de datos y obtener VideoId
            using (var scope = _scopeFactory.CreateScope())
            {
                var videoService = scope.ServiceProvider.GetRequiredService<IVideoService>();

                _currentVideo = await videoService.AddVideoAsync(_currentVideo);

                // Verificar que el VideoId se haya generado
                if (_currentVideo.VideoId == 0)
                {
                    throw new InvalidOperationException("El VideoId no se ha generado correctamente.");
                }

                // Crear log de inicio de grabación con VideoId
                await CreateLogAsync("Inicio de grabación");
            }

            // Reset `_hasLoggedPersonDetected` para permitir la creación de logs en futuras detecciones
            _hasLoggedPersonDetected = false;

            // Iniciar el timer de 60 segundos
            _intervalTimer.Start();
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
            _videoWriter?.Dispose();
            _intervalTimer?.Stop();
            _intervalTimer = null;
        }
    }
}
