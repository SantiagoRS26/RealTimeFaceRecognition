using BLL.Interfaces;
using DAL.Context;
using DAL.Interfaces;
using Models.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Services
{
    public class VideoService : IVideoService
    {
        private readonly IUnitOfWork _unitOfWork;

        public VideoService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
        }

        public async Task<IEnumerable<Video>> GetAllVideosAsync()
        {
            return await _unitOfWork.Videos.GetAllAsync();
        }

        public async Task<Video> GetVideoByIdAsync(int videoId)
        {
            return await _unitOfWork.Videos.GetByIdAsync(videoId);
        }

        public async Task<Video> AddVideoAsync(Video video)
        {
            if (video == null)
                throw new ArgumentNullException(nameof(video));

            try
            {
                await _unitOfWork.Videos.AddAsync(video);
                await _unitOfWork.CommitAsync();

                if (video.VideoId == 0)
                {
                    throw new InvalidOperationException("El VideoId no se generó correctamente al guardar el Video.");
                }

                return video;
            }
            catch (Exception ex)
            {
                // Registrar o lanzar la excepción para depurar
                Console.WriteLine($"Error en AddVideoAsync: {ex.Message}");
                throw;
            }
        }


        public async Task UpdateVideoAsync(Video video)
        {
            if (video == null)
                throw new ArgumentNullException(nameof(video));

            _unitOfWork.Videos.Update(video);
            await _unitOfWork.CommitAsync();
        }

        public async Task DeleteVideoAsync(int videoId)
        {
            var video = await _unitOfWork.Videos.GetByIdAsync(videoId);
            if (video == null)
                throw new KeyNotFoundException($"Video with ID {videoId} not found.");

            _unitOfWork.Videos.Remove(video);
            await _unitOfWork.CommitAsync();
        }
    }
}
