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

            var mainWindow = ServiceProvider.GetRequiredService<MainWindow>();
            mainWindow.Show();
        }

        private void ConfigureServices(ServiceCollection services)
        {
            var awsAccessKey = ConfigurationManager.AppSettings["AWS:AccessKey"];
            var awsSecretKey = ConfigurationManager.AppSettings["AWS:SecretKey"];
            var awsRegion = ConfigurationManager.AppSettings["AWS:Region"];
            var awsBucketName = ConfigurationManager.AppSettings["AWS:S3BucketName"];

            // Configure DbContext as Transient
            services.AddTransient<ApplicationDbContext>(provider =>
            {
                var optionsBuilder = new DbContextOptionsBuilder<ApplicationDbContext>();
                var connectionString = ConfigurationManager.ConnectionStrings["DefaultConnection"].ConnectionString;
                optionsBuilder.UseNpgsql(connectionString);
                return new ApplicationDbContext(optionsBuilder.Options);
            });

            // Register repositories and UnitOfWork as Transient
            services.AddTransient(typeof(IGenericRepository<>), typeof(GenericRepository<>));
            services.AddTransient<IUnitOfWork, UnitOfWork>();

            // Register BLL services as Transient
            services.AddTransient<IVideoService, VideoService>();
            services.AddTransient<ILogService, LogService>();

            // Register services for video capture and detection
            services.AddSingleton<IVideoCaptureService, VideoCaptureService>();
            services.AddTransient<IS3Service>(provider =>
            {
                return new S3Service(awsAccessKey, awsSecretKey, awsRegion, awsBucketName);
            });
            services.AddSingleton<IFaceModelLoader, FaceModelLoader>();
            services.AddSingleton<IFaceDetectionService>(provider =>
            {
                var modelLoader = provider.GetService<IFaceModelLoader>();
                string modelConfiguration = "Models/deploy.prototxt";
                string modelWeights = "Models/res10_300x300_ssd_iter_140000.caffemodel";
                float confThreshold = 0.8f;

                return new FaceDetectionDNNService(modelLoader, modelConfiguration, modelWeights, confThreshold);
            });

            // Register ViewModels and Windows
            services.AddSingleton<MainViewModel>();
            services.AddSingleton<MainWindow>();
        }
    }
}