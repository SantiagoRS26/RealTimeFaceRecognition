using Models.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Interfaces
{
    public interface IVideoService
    {
        Task<IEnumerable<Video>> GetAllVideosAsync();
        Task<Video> GetVideoByIdAsync(int videoId);
        Task<Video> AddVideoAsync(Video video);
        Task UpdateVideoAsync(Video video);
        Task DeleteVideoAsync(int videoId);
    }
}
