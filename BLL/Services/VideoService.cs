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
        private readonly IGenericRepository<Video> _videoRepository;

        public VideoService(IGenericRepository<Video> videoRepository)
        {
            _videoRepository = videoRepository;
        }

        public async Task<IEnumerable<Video>> GetAllVideosAsync()
        {
            return await _videoRepository.GetAllAsync();
        }

        public async Task<Video> GetVideoByIdAsync(int videoId)
        {
            return await _videoRepository.GetByIdAsync(videoId);
        }

        public async Task AddVideoAsync(Video video)
        {
            await _videoRepository.AddAsync(video);
            await _videoRepository.SaveChangesAsync();
        }

        public async Task UpdateVideoAsync(Video video)
        {
            _videoRepository.Update(video);
            await _videoRepository.SaveChangesAsync();
        }

        public async Task DeleteVideoAsync(int videoId)
        {
            var video = await _videoRepository.GetByIdAsync(videoId);
            if (video != null)
            {
                _videoRepository.Remove(video);
                await _videoRepository.SaveChangesAsync();
            }
        }
    }
}
