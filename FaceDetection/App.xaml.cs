using BLL.Interfaces;
using BLL.Services;
using DAL.Context;
using DAL.Interfaces;
using DAL.Repositories;
using DAL.Servicios;
using FaceDetection.ViewModels;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using System.Configuration;
using System.Data;
using System.Windows;

namespace FaceDetection
{
    public partial class App : Application
    {
        public static IServiceProvider ServiceProvider { get; private set; }

        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            var services = new ServiceCollection();
            ConfigureServices(services);
            ServiceProvider = services.BuildServiceProvider();

            // Crear un scope para resolver servicios Scoped
            using (var scope = ServiceProvider.CreateScope())
            {
                var mainWindow = scope.ServiceProvider.GetRequiredService<MainWindow>();
                mainWindow.Show();
            }
        }

        private void ConfigureServices(ServiceCollection services)
        {
            services.AddDbContext<ApplicationDbContext>(options =>
            {
                var connectionString = ConfigurationManager.ConnectionStrings["DefaultConnection"].ConnectionString;
                options.UseNpgsql(connectionString);
            });

            // Repositorios Genéricos
            services.AddScoped(typeof(IGenericRepository<>), typeof(GenericRepository<>));

            // Servicios BLL
            services.AddScoped<IVideoService, VideoService>();
            services.AddScoped<IDetectionService, DetectionService>();
            services.AddScoped<IIntervalService, IntervalService>();
            services.AddScoped<ILogService, LogService>();
            // Otros servicios BLL si existen

            // Servicios para Captura de Video y Detección
            services.AddSingleton<IVideoCaptureService, VideoCaptureService>();
            services.AddSingleton<IFaceModelLoader, FaceModelLoader>();
            services.AddSingleton<IFaceDetectionService>(provider =>
            {
                var modelLoader = provider.GetService<IFaceModelLoader>();
                string modelConfiguration = "Models/deploy.prototxt";
                string modelWeights = "Models/res10_300x300_ssd_iter_140000.caffemodel";
                float confThreshold = 0.5f;

                return new FaceDetectionDNNService(modelLoader, modelConfiguration, modelWeights, confThreshold);
            });

            // ViewModels y Windows
            services.AddScoped<MainViewModel>();
            services.AddScoped<MainWindow>();
        }
    }
}
