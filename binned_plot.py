import matplotlib.pyplot
import numpy


def binned_plot(x, y, bins=20, kind='quantiles', weights=None,
                xlabel=None, ylabel=None, include_regression=True,
                with_intercept=True):
    '''
    Binned plot of `y` vs `x`.
    '''
    x = numpy.asarray(x).ravel()
    y = numpy.asarray(y).ravel()

    if len(x) != len(y):
        raise ValueError('x and y arrays must have the same lengths')

    mask = numpy.isfinite(x) & numpy.isfinite(y)
    x = x[mask]
    y = y[mask]

    if kind == 'quantiles':
        q = numpy.linspace(0, 100, bins + 1)[1:-1]
        boundaries = numpy.percentile(x, q)
    elif kind == 'uniform':
        boundaries = numpy.linspace(x.min(), x.max(), bins + 1)[1:-1]
    else:
        raise ValueError('unrecognized binning approach %r' % kind)

    bin_ix = numpy.digitize(x, boundaries)

    count_B  = numpy.bincount(bin_ix, weights=weights, minlength=bins)
    b_mask_B = count_B > 1

    if weights is None:
        weights = numpy.ones_like(x)

    count_b = count_B[b_mask_B]
    xsum_b  = numpy.bincount(bin_ix, weights * x,     minlength=bins)[b_mask_B]
    ysum_b  = numpy.bincount(bin_ix, weights * y,     minlength=bins)[b_mask_B]
    x2sum_b = numpy.bincount(bin_ix, weights * x * x, minlength=bins)[b_mask_B]
    y2sum_b = numpy.bincount(bin_ix, weights * y * y, minlength=bins)[b_mask_B]

    xmean_b = xsum_b / count_b
    ymean_b = ysum_b / count_b
    xerr_b  = numpy.sqrt(numpy.fmax(x2sum_b / count_b - xmean_b ** 2, 0) / count_b)
    yerr_b  = numpy.sqrt(numpy.fmax(y2sum_b / count_b - ymean_b ** 2, 0) / count_b)

    matplotlib.pyplot.errorbar(xmean_b, ymean_b, xerr=xerr_b, yerr=yerr_b,
                               linestyle='', capsize=3)

    xmean = numpy.average(x, weights=weights)
    ymean = numpy.average(y, weights=weights)
    dx = x - xmean
    dy = y - ymean

    dwx = dx * weights
    dwy = dy * weights

    corr  = (dwx @ dy) / numpy.sqrt((dwx @ dx) * (dwy @ dy))

    ax = matplotlib.pyplot.gca()

    if include_regression:
        if with_intercept:
            beta  = (dwx @ dy) / (dwx @ dx)
            alpha = ymean - beta * xmean
            if alpha < 0:
                eq = 'y = %.5g * x - %.5g' % (beta, -alpha)
            else:
                eq = 'y = %.5g * x + %.5g' % (beta, alpha)
        else:
            wx = x * weights
            beta = (wx @ y) / (wx @ x)
            alpha = 0
            eq = 'y = %.5g * x' % beta

        eq += ' (rho=%.3f)' % corr

        ylim = ax.get_ylim()
        (x0, x1) = ax.get_xlim()
        matplotlib.pyplot.plot([x0, x1], [alpha + beta * x0, alpha + beta * x1], 'r')
        ax.set_xlim(x0, x1)
        ax.set_ylim(ylim)
    else:
        eq = 'rho=%.3f' % corr

    if xlabel is not None:
        ax.set_xlabel('%s\n%s' % (xlabel, eq))
    else:
        ax.set_xlabel(eq)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(True)