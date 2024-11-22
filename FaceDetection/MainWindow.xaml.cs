using BLL.Interfaces;
using Emgu.CV;
using Emgu.CV.Ocl;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using FaceDetection.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using System.Drawing;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace FaceDetection
{
    public partial class MainWindow : Window
    {
        private readonly MainViewModel _viewModel;

        public MainWindow()
        {
            InitializeComponent();
            _viewModel = App.ServiceProvider.GetService<MainViewModel>();
            DataContext = _viewModel;
        }

        protected override void OnClosed(EventArgs e)
        {
            if (_viewModel is IDisposable disposable)
            {
                disposable.Dispose();
            }
            base.OnClosed(e);
        }
    }
}