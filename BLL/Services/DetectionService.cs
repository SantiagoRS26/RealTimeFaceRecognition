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
    public class DetectionService : IDetectionService
    {
        private readonly IGenericRepository<Detection> _detectionRepository;

        public DetectionService(IGenericRepository<Detection> detectionRepository)
        {
            _detectionRepository = detectionRepository;
        }

        public async Task<IEnumerable<Detection>> GetDetectionsByIntervalAsync(int intervalId)
        {
            return await _detectionRepository.GetAsync(d => d.IntervalId == intervalId);
        }

        public async Task<Detection> GetDetectionByIdAsync(int detectionId)
        {
            return await _detectionRepository.GetByIdAsync(detectionId);
        }

        public async Task AddDetectionAsync(Detection detection)
        {
            await _detectionRepository.AddAsync(detection);
            await _detectionRepository.SaveChangesAsync();
        }

        public async Task UpdateDetectionAsync(Detection detection)
        {
            _detectionRepository.Update(detection);
            await _detectionRepository.SaveChangesAsync();
        }

        public async Task DeleteDetectionAsync(int detectionId)
        {
            var detection = await _detectionRepository.GetByIdAsync(detectionId);
            if (detection != null)
            {
                _detectionRepository.Remove(detection);
                await _detectionRepository.SaveChangesAsync();
            }
        }
    }
}
