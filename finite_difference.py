def central_difference(f, x, h=1e-2, order=1, precision = 6):
    if order == 1:
        if precision == 2:
            return (f(x + h) - f(x - h)) / (h)
        elif precision == 6:
            return (float(1/60)*f(x + 3*h) - float(3/20)*f(x + 2*h) + 0.75*f(x + h) - 0.75*f(x - h) + float(3/20)*f(x - 2*h) - float(1/60)*f(x - 3*h))/(h)
    elif order == 2:
        if precision == 2:
            return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
        elif precision == 6:
            return (1/90*f(x+3*h) - 3/20*f(x+2*h) + 1.5*f(x+h) - 49/18*f(x) + 1.5*f(x-h) - 3/20*f(x-2*h) + 1/90*f(x- 3*h))/(h**2)
            #return (-float(1/56)*f(x + 4*h) + float(8/315)*f(x+ 3*h) - 1/5* f(x + 2*h) + 8/5*f(x+h) - 205/72*f(x) + 8/5*f(x-h) - 1/5*f(x-2*h) + 8/315*f(x-3*h) - 1/560*f(x-4*h))/(h**2)
    elif order == 3:
        return (0.5*f(x + 2*h) - f(x + h) + f(x - h) - 0.5*f(x - 2*h))/(h**3)