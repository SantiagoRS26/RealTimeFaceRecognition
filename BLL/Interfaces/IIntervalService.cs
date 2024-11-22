using Models.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Interfaces
{
    public interface IIntervalService
    {
        Task<IEnumerable<Interval>> GetAllIntervalsAsync();
        Task<Interval> GetIntervalByIdAsync(int intervalId);
        Task<Interval> GetIntervalWithDetailsAsync(int intervalId);
        Task AddIntervalAsync(Interval interval);
        Task UpdateIntervalAsync(Interval interval);
        Task DeleteIntervalAsync(int intervalId);
    }
}
