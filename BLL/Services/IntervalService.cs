using BLL.Interfaces;
using DAL.Interfaces;
using Models.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Services
{
    public class IntervalService : IIntervalService
    {
        private readonly IGenericRepository<Interval> _intervalRepository;

        public IntervalService(IGenericRepository<Interval> intervalRepository)
        {
            _intervalRepository = intervalRepository;
        }

        public async Task<IEnumerable<Interval>> GetAllIntervalsAsync()
        {
            return await _intervalRepository.GetAllAsync();
        }

        public async Task<Interval> GetIntervalByIdAsync(int intervalId)
        {
            return await _intervalRepository.GetByIdAsync(intervalId);
        }

        public async Task<Interval> GetIntervalWithDetailsAsync(int intervalId)
        {
            var intervals = await _intervalRepository.GetAsync(i => i.IntervalId == intervalId);
            return intervals.FirstOrDefault();
        }

        public async Task AddIntervalAsync(Interval interval)
        {
            await _intervalRepository.AddAsync(interval);
            await _intervalRepository.SaveChangesAsync();
        }

        public async Task UpdateIntervalAsync(Interval interval)
        {
            _intervalRepository.Update(interval);
            await _intervalRepository.SaveChangesAsync();
        }

        public async Task DeleteIntervalAsync(int intervalId)
        {
            var interval = await _intervalRepository.GetByIdAsync(intervalId);
            if (interval != null)
            {
                _intervalRepository.Remove(interval);
                await _intervalRepository.SaveChangesAsync();
            }
        }
    }
}
