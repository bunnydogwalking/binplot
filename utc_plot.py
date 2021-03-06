import matplotlib.dates
import matplotlib.pyplot
import numpy
import pandas


def utc_plot(utcs, y, tz='America/New_York', include_date=None,
             include_time=True, ax=None, **kwargs):
    grid = kwargs.pop('grid', True)

    utcs = numpy.asarray(utcs)
    y    = numpy.asarray(y)

    if utcs.dtype != 'datetime64[ns]':
        utcs = numpy.array(utcs * 1_000_000_000, dtype='datetime64[ns]')

    if include_date is None:
        include_date = utcs.max() - utcs.min() > numpy.timedelta64(1, 'D')

    # If you have dates instead of utcs, use something like:
    #
    #   dates = pandas.DatetimeIndex([date for date in dates])
    #
    # where each date should be a datetime.date object.
    utcs = pandas.DatetimeIndex(utcs).tz_localize('UTC').tz_convert(tz)

    if ax is None:
        ax = matplotlib.pyplot.gca()

    # Make pandas timestamps plottable
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    lines = ax.plot(utcs, y, **kwargs)

    if grid:
        ax.grid(True)

    if include_time:
        # XXX These formatters do not adjust to zoom level
        xaxis_fmt_str = '%H:%M:%S'
        if include_date:
            xaxis_fmt_str = '%Y-%m-%d ' + xaxis_fmt_str

        date_fmt = matplotlib.dates.DateFormatter(xaxis_fmt_str, tz=utcs.tz)
        ax.xaxis.set_major_formatter(date_fmt)

        # Include microseconds in the mouse-over display
        ax.fmt_xdata = matplotlib.dates.DateFormatter(xaxis_fmt_str + '.%f', tz=utcs.tz)
    else:
        date_locator   = matplotlib.dates.AutoDateLocator()
        date_formatter = matplotlib.dates.AutoDateFormatter(date_locator)

        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)

    ax.xaxis.set_tick_params(rotation=30)

    return lines
