import pybaselines 


def get_algo_instance(Alg,Alg_X, Alg_Data, Alg_Fit_Order, Alg_Num_Iter, Alg_Weight):
    # def gaussian(x, a1, x1, sigma1,a2, x2, sigma2,b0):
    #     return a1 * np.exp(-(x - x1)**2 / (2 * sigma1**2)) + a2 * np.exp(-(x - x2)**2 / (2 * sigma2**2))  + b0
    # if Alg == 'Gaussian':
    #     popt, pcov = curve_fit(gaussian, Alg_X[Alg_Weight], Alg_Data[Alg_Weight], p0=[0.08, -0.5, 1,0.08,0,1,0], maxfev=5000)
    #     return gaussian( Alg_X, *popt)
    # if Alg == 'SACMES':
    #     def biexponential_func(t, a, b, c, d, e):

    #         return a * np.exp(-t * b) + c * np.exp(-t * d) + e
    #     try:
    #         popt, pcov =   curve_fit( biexponential_func,Alg_X[Alg_Weight],Alg_Data[Alg_Weight]) 
    #         # print(popt)
    #         return ((  biexponential_func(Alg_X,*popt), pcov  ), None )
    #     except Exception as e:
    #         return (0, 0), str(e)    
    #region polynomial  
    if Alg == 'goldindec':
        try:
            return ( (  pybaselines.polynomial.goldindec(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'imodpoly':
        try:
            return ( (  pybaselines.polynomial.imodpoly(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'imodpoly4':
        try:
            return ( (  pybaselines.polynomial.imodpoly(Alg_Data, poly_order=4, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'loess':
        try:
            return ( (  pybaselines.polynomial.loess(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'modpoly':
        try:
            return ( (  pybaselines.polynomial.modpoly(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'penalized_poly':
        try:
            return ( (  pybaselines.polynomial.penalized_poly(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'penalized_poly4':
        try:
            return ( (  pybaselines.polynomial.penalized_poly(Alg_Data, poly_order=4, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'poly':
        try:
            return ( (  pybaselines.polynomial.poly(Alg_Data, poly_order=Alg_Fit_Order, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'quant_reg':
        try:
            return ( (  pybaselines.polynomial.quant_reg(Alg_Data, poly_order=Alg_Fit_Order, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion
    # region spline
    elif Alg == 'corner_cutting':
        try:
            return ( (  pybaselines.spline.corner_cutting(Alg_Data, max_iter=Alg_Num_Iter) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'irsqr':
        try:
            return ( (  pybaselines.spline.irsqr(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'mixture_model':
        try:
            return ( (  pybaselines.spline.mixture_model(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_airpls':
        try:
            return ( (  pybaselines.spline.pspline_airpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_arpls':
        try:
            return ( (  pybaselines.spline.pspline_arpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_asls':
        try:
            return ( (  pybaselines.spline.pspline_asls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_aspls':
        try:
            return ( (  pybaselines.spline.pspline_aspls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_derpsalsa':
        try:
            return ( (  pybaselines.spline.pspline_derpsalsa(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_drpls':
        try:
            return ( (  pybaselines.spline.pspline_drpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_iarpls':
        try:
            return ( (  pybaselines.spline.pspline_iarpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_iasls':
        try:
            return ( (  pybaselines.spline.pspline_iasls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_mpls':
        try:
            return ( (  pybaselines.spline.pspline_mpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'pspline_psalsa':
        try:
            return ( (  pybaselines.spline.pspline_psalsa(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion

    # region whittaker
    elif Alg == 'airpls':
        try:
            return ( (  pybaselines.whittaker.airpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'arpls':
        try:
            return ( (  pybaselines.whittaker.arpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'asls':
        try:
            return ( (  pybaselines.whittaker.asls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'aspls':
        try:
            return ( (  pybaselines.whittaker.aspls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'derpsalsa':
        try:
            return ( (  pybaselines.whittaker.derpsalsa(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'drpls':
        try:
            return ( (  pybaselines.whittaker.drpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'iarpls':
        try:
            return ( (  pybaselines.whittaker.iarpls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'iasls':
        try:
            return ( (  pybaselines.whittaker.iasls(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'psalsa':
        try:
            return ( (  pybaselines.whittaker.psalsa(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight, x_data=Alg_X) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion

    # region classification
    elif Alg == 'cwt_br':
        try:
            return ( (  pybaselines.classification.cwt_br(Alg_Data, poly_order=3, min_length=2, max_iter=Alg_Num_Iter, tol=0.001, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'dietrich':
        try:
            return ( (  pybaselines.classification.dietrich(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'fabc':
        try:
            return ( (  pybaselines.classification.fabc(Alg_Data, weights=Alg_Weight, weights_as_mask=True) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'fastchrom':
        try:
            return ( (  pybaselines.classification.fastchrom(Alg_Data, max_iter=Alg_Num_Iter, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'golotvin':
        try:
            return ( (  pybaselines.classification.golotvin(Alg_Data, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'rubberband':
        try:
            return ( (  pybaselines.classification.rubberband(Alg_Data, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    elif Alg == 'std_distribution':
        try:
            return ( (  pybaselines.classification.std_distribution(Alg_Data, weights=Alg_Weight) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion

    # region misc
    elif Alg == 'beads':
        try:
            return ( (  pybaselines.misc.beads(Alg_Data, max_iter=Alg_Num_Iter) ), None )
        except Exception as e:
            return (0, 0), str(e)
    # endregion
    else:
        raise ValueError(f"Unknown Algorithm: {Alg}")
    
