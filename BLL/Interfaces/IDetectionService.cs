using Models.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Interfaces
{
    public interface IDetectionService
    {
        Task<IEnumerable<Detection>> GetDetectionsByIntervalAsync(int intervalId);
        Task<Detection> GetDetectionByIdAsync(int detectionId);
        Task AddDetectionAsync(Detection detection);
        Task UpdateDetectionAsync(Detection detection);
        Task DeleteDetectionAsync(int detectionId);
    }
}
