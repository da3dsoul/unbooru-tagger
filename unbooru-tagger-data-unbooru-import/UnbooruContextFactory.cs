using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;
using unbooru.Abstractions.Interfaces;
using unbooru.Core;

namespace UnbooruTagger.Data;

/// <summary>
/// Constructs a real <c>unbooru.Core.CoreContext</c> against a given connection
/// string, without needing unbooru's full DI/hosting setup or its
/// JsonSettingsProvider's file-path convention — CoreContext.OnConfiguring does all
/// the SQL Server wiring itself once given a connection string via
/// <see cref="ISettingsProvider{T}"/>.
/// </summary>
public static class UnbooruContextFactory
{
    public static CoreContext Create(string connectionString) =>
        new(new DbContextOptions<CoreContext>(), new StaticSettingsProvider(connectionString), NullLogger<CoreContext>.Instance);

    private sealed class StaticSettingsProvider(string connectionString) : ISettingsProvider<CoreSettings>
    {
        public TResult Get<TResult>(Func<CoreSettings, TResult> func) => func(new CoreSettings { ConnectionString = connectionString });

        public void Update(Action<CoreSettings> func) =>
            throw new NotSupportedException("This data pipeline only reads from unbooru's database.");
    }
}
