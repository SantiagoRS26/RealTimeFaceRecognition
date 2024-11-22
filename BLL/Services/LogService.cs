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
    public class LogService : ILogService
    {
        private readonly IGenericRepository<Log> _logRepository;

        public LogService(IGenericRepository<Log> logRepository)
        {
            _logRepository = logRepository;
        }

        public async Task<IEnumerable<Log>> GetAllLogsAsync()
        {
            return await _logRepository.GetAllAsync();
        }

        public async Task<Log> GetLogByIdAsync(int logId)
        {
            return await _logRepository.GetByIdAsync(logId);
        }

        public async Task<IEnumerable<Log>> GetLogsByVideoIdAsync(int videoId)
        {
            return await _logRepository.GetAsync(l => l.VideoId == videoId);
        }

        public async Task<IEnumerable<Log>> GetLogsByIntervalIdAsync(int intervalId)
        {
            return await _logRepository.GetAsync(l => l.IntervalId == intervalId);
        }

        public async Task AddLogAsync(Log log)
        {
            await _logRepository.AddAsync(log);
            await _logRepository.SaveChangesAsync();
        }

        public async Task UpdateLogAsync(Log log)
        {
            _logRepository.Update(log);
            await _logRepository.SaveChangesAsync();
        }

        public async Task DeleteLogAsync(int logId)
        {
            var log = await _logRepository.GetByIdAsync(logId);
            if (log != null)
            {
                _logRepository.Remove(log);
                await _logRepository.SaveChangesAsync();
            }
        }
    }
}
