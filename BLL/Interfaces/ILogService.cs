using Models.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Interfaces
{
    public interface ILogService
    {
        Task<IEnumerable<Log>> GetAllLogsAsync();
        Task<Log> GetLogByIdAsync(int logId);
        Task<IEnumerable<Log>> GetLogsByVideoIdAsync(int videoId);
        Task AddLogAsync(Log log);
        Task UpdateLogAsync(Log log);
        Task DeleteLogAsync(int logId);
    }
}
