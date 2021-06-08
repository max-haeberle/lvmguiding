import lvmguiding
import numpy as np
import os

import time
from astropy.coordinates import SkyCoord
from importlib import reload

import healpy as hp
from astropy.io import fits
import astropy.units as u
import scipy.spatial


import sys
sys.path.insert(1, 'lvmifusim/')
import IFU

irdc_data = np.loadtxt("target_db_run072021a_v2.tbl",skiprows=1)


def check_cloud(irdc_index,width=1,n=240,verbose=False):
    
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib
    from matplotlib.patches import Ellipse
    
    irdc_identifier = int(irdc_data[irdc_index,0])

    ra_irdc = irdc_data[irdc_index,1]
    dec_irdc = irdc_data[irdc_index,2]


    if verbose: print("Identifier: ",irdc_identifier)
    if verbose: print("Coordinates: ",ra_irdc,dec_irdc)
    if verbose: print("Radius: ",irdc_data[irdc_index,7],"arcsec = ",irdc_data[irdc_index,7]/3600," deg")

    irdc_r = irdc_data[irdc_index,7] / 3600

    c_cloud = SkyCoord(ra_irdc,dec_irdc,unit="deg")
    
    y = np.linspace(-0.5*width, 0.5*width,n)

    x = np.linspace(-0.5/np.cos(np.deg2rad(dec_irdc))*width,0.5/np.cos(np.deg2rad(dec_irdc))*width,n)

    xx, yy = np.meshgrid(x, y, sparse=False)


    ras2 = ra_irdc+ xx.flatten()
    decs2 = dec_irdc + yy.flatten()

    ras2u = ras2*u.deg
    decs2u= decs2*u.deg

    coordinate_list = SkyCoord(ras2u,decs2u)
    #coordinate_list = []
    #for i in range(len(ras2)):
    #    coordinate_list.append(SkyCoord(ras2[i],decs2[i],unit="deg"))

    if verbose: print("I will check {} different pointings.".format(len(coordinate_list)))
    
    my_instrument = lvmguiding.InstrumentParameters()
    my_instrument.outer_search_radius = 1.1*np.sqrt(2)*width/2
    my_instrument.mag_lim_lower = 21 #If you use a value higher than 17 here, make sure to remote-query the Gaia catalog
    #remote_query = True


    t0 = time.time()
    data_combined = lvmguiding.get_cat_using_healpix2(c_cloud,plotflag=False,inst=my_instrument)
    #ras,decs,dd_x_mm,dd_y_mm,chip_xxs,chip_yys,mags,culled_cat = lvmguiding.find_guide_stars(c_cloud,pa=0,recycled_cat=data_combined)
    t1 = time.time()
    if verbose: print()
    if verbose: print("Duration using healpix preselection: {:.4f} s".format(t1-t0))


    #t0 = time.time()
    #data_combined = get_cat_using_healpix(c)
    #ras,decs,dd_x_mm,dd_y_mm,chip_xxs,chip_yys,mags,culled_cat = lvmguiding.find_guide_stars(c_cloud,pa=0,plotflag=False,recycled_cat=culled_cat)#,inst=my_instrument
    #t1 = time.time()
    #if verbose: print()
    #if verbose: print("Duration using preselected Gaia cat: {:.4f} s".format(t1-t0))




    t0 = time.time()
    #data_combined = get_cat_using_healpix(c)
    dd_x_mm,dd_y_mm,culled_cat2= lvmguiding.find_guide_stars(c_cloud,pa=0,plotflag=False,recycled_cat=data_combined,return_focal_plane_coords=True,inst=my_instrument)
    t1 = time.time()
    if verbose: print()
    if verbose: print("Duration using preselected Gaia and only doing conversion to coordinates: {:.4f} s".format(t1-t0))
    if verbose: print("There are {} sources in the catalog".format(len(culled_cat2)))

    # Define the IFU, it has to be doubled checked whether the used lens radius is alright and whether the IFU lib produces the correct results

    my_ifu = IFU.IFU(4) 
    lens_radii = 0.315/2 * np.ones_like(my_ifu.lensx) #This comes from PDR Document, Figure 4
    
    ifu_point_list = []

    for i in range(len(my_ifu.lensx)):
        current_x = my_ifu.lensx[i]
        current_y = my_ifu.lensy[i]

        ifu_point_list.append((current_x,current_y))

        
        plotflag= True

    dd_x_mm,dd_y_mm,culled_cat= lvmguiding.find_guide_stars(c_cloud,pa=0,plotflag=False,recycled_cat=culled_cat2,return_focal_plane_coords=True,inst=my_instrument)

    nmax=10
    data_dict={}
    data_dict["x"] = dd_x_mm#[selection_mag]#.filled() 
    data_dict["y"] = dd_y_mm#[selection_mag]#.filled() 
    data_dict["m"] = culled_cat["phot_g_mean_mag"]#[selection_mag]
    m_array = np.concatenate((np.array(data_dict["m"]),np.array([float("nan")])))
    #if verbose: print("Creating Tree...")
    t0 = time.time()
    #xy_list = np.vstack((dd_x_mm[selection_mag].filled(),dd_y_mm[selection_mag].filled())).T
    xy_list = np.vstack((data_dict["x"],data_dict["y"])).T
    YourTreeName = scipy.spatial.cKDTree(xy_list, leafsize=100)
    t1 = time.time()
    #if verbose: print("... done! Time: {:.4f} s".format(t1-t0))



    query_result = YourTreeName.query(ifu_point_list, k=nmax, distance_upper_bound=lens_radii[0])#

    contaminated = np.isfinite(query_result[0][:,0])

    current_r = lens_radii[0]

    if plotflag:
        fig,(ax1,ax2) = plt.subplots(figsize=(20,12),ncols=2)
        fig.suptitle("Cloud ID: {}\nPointing Coordinates:\nRA: {:.6f}\nDEC: {:.6f}\n".format(irdc_identifier,c_cloud.ra,c_cloud.dec))
        ax1.set_title("Full focal plane\nN Sources: {}".format(len(dd_x_mm)))


        ax1.set_xlabel("Focal plane x [mm]")
        ax1.set_ylabel("Focal plane y [mm]")
        ax2.set_xlabel("Focal plane x [mm]")
        ax2.set_ylabel("Focal plane y [mm]")

        ax1.plot(dd_x_mm,dd_y_mm,"ko",ms=1)
        ax1.plot(my_ifu.lensx,my_ifu.lensy,"bo",ms=1)
        ax1.plot(np.array(my_ifu.lensx)[contaminated],np.array(my_ifu.lensy)[contaminated],"ro",ms=1)
        ax1.set_aspect("equal")


        ax2.set_xlim(-8,8)
        ax2.set_ylim(-8,8)
        ax2.set_aspect("equal")

        patches = [plt.Circle(center, size) for center, size in zip(np.stack((my_ifu.lensx,my_ifu.lensy),axis=1),lens_radii)]

        patches_contaminated = [plt.Circle(center,size) for center, size in zip(np.stack((np.array(my_ifu.lensx)[contaminated],np.array(my_ifu.lensy)[contaminated]),axis=1),lens_radii[contaminated])]


        #
        coll = matplotlib.collections.PatchCollection(patches, facecolors='none',edgecolor="b")
        coll2 = matplotlib.collections.PatchCollection(patches_contaminated, facecolors='r',alpha=0.5)
        ax2.add_collection(coll)
        ax2.add_collection(coll2)

        ax2.plot(dd_x_mm,dd_y_mm,"ko",ms=2)
        #ax2.plot(dd_x_mm[selection_mag],dd_y_mm[selection_mag],"ko",ms=4)


        ax2.set_title("IFU\n{} of {} ({:.1f}%) fibers are contaminated with stars brighter {} gmag".format(np.sum(contaminated),len(contaminated),100*np.sum(contaminated)/len(contaminated),99))

        circle = plt.Circle((-7,7),current_r,facecolor="none",edgecolor="b",alpha=0.5)
        ax2.add_patch(circle)
        circle = plt.Circle((-7,6.5),current_r,facecolor="r",alpha=0.5)
        ax2.add_patch(circle)

        #circle = plt.Circle((0,0),1.4,facecolor="r",alpha=0.5)
        #ax2.add_patch(circle)

        ax2.text(-6.5,7,"Star-free Fiber")#.format(maglim))
        ax2.text(-6.5,6.5,"Contaminated Fiber")#.format(maglim)

    fig.savefig("irdc_results2/{:09d}_fiber_view.png".format(irdc_identifier),dpi=200,bbox_inches="tight",facecolor='white', transparent=False)
    

    
    ratios = []

    combined_neighbour_array = []

    my_instrument2 = my_instrument
    my_instrument2.upper_search_radius = 0.1

    for index,c in enumerate(coordinate_list):
        dd_x_mm,dd_y_mm,culled_cat= lvmguiding.find_guide_stars(c,pa=0,plotflag=False,recycled_cat=culled_cat2,return_focal_plane_coords=True,inst=my_instrument2)

        nmax=10
        data_dict={}
        data_dict["x"] = dd_x_mm#[selection_mag]#.filled() 
        data_dict["y"] = dd_y_mm#[selection_mag]#.filled() 
        data_dict["m"] = culled_cat["phot_g_mean_mag"]#[selection_mag]
        m_array = np.concatenate((np.array(data_dict["m"]),np.array([float("nan")])))
        #if verbose: print("Creating Tree...")
        t0 = time.time()
        #xy_list = np.vstack((dd_x_mm[selection_mag].filled(),dd_y_mm[selection_mag].filled())).T
        xy_list = np.vstack((data_dict["x"],data_dict["y"])).T
        YourTreeName = scipy.spatial.cKDTree(xy_list, leafsize=100)
        t1 = time.time()
        #if verbose: print("... done! Time: {:.4f} s".format(t1-t0))



        query_result = YourTreeName.query(ifu_point_list, k=nmax, distance_upper_bound=lens_radii[0])#

        contaminated = np.isfinite(query_result[0][:,0])

        m_neighbours = m_array[query_result[1]]
        flux_neighbours = 10**(-0.4*m_neighbours)
        flux_combined = np.nansum(flux_neighbours,axis=1)
        m_combined = -2.5*np.log10(flux_combined)

        combined_neighbour_array.append(m_combined)

        #if verbose: print("Ratio of contaminated pixels: ",np.sum(contaminated)/len(contaminated))
        ratios.append(np.sum(contaminated)/len(contaminated))

        if verbose: print(index,len(coordinate_list),end="\r")

    if verbose: print()
    
    min_mags = np.arange(10,22,1)

    fig,axarr = plt.subplots(figsize=(20,20),ncols=4,nrows=3)

    fig.suptitle("Cloud ID: {}\nPointing Coordinates:\nRA: {:.6f}\nDEC: {:.6f}\n".format(irdc_identifier,c_cloud.ra,c_cloud.dec))


    outfile = open("irdc_results2/{:09d}_contamination.txt".format(irdc_identifier),"w")


    for index,ax in enumerate(axarr.flatten()):
        mag_lim = min_mags[index]







        image = np.reshape(np.sum(np.array(combined_neighbour_array)<mag_lim,axis=1),xx.shape)/len(ifu_point_list)

        indices_in_radius = (xx*np.cos(np.deg2rad(dec_irdc)))**2+yy**2 < (99*irdc_r)**2

        min_index = np.argmin(image[indices_in_radius])

        best_cont = image[indices_in_radius][min_index]
        best_xx = xx[indices_in_radius][min_index]
        best_yy = yy[indices_in_radius][min_index]
        best_ra = coordinate_list.ra.value[indices_in_radius.flatten()][min_index]
        best_dec = coordinate_list.dec.value[indices_in_radius.flatten()][min_index]
        best_c = image[indices_in_radius][min_index]

        if verbose: print("Maglim: {:12.6f} Cont: {:12.6f} Best RA: {:12.6f} Best DEC: {:12.6f} Best dx: {:12.6f} Best dy: {:12.6f}".format(mag_lim,best_cont,best_ra,best_dec,best_xx,best_yy))
        outfile.write("{:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n".format(mag_lim,best_cont,best_ra,best_dec,best_xx,best_yy))
        ax.set_title("Maglim: "+str(mag_lim)+"\nBest Coordinates (Contamination = {:.2f})\nRA: {:.6f} (dx={:.6f})\nDEC: {:.6f} (dy={:.6f})\n".format(best_cont,best_ra,best_xx,best_dec,best_yy))
        #image[indices_in_radius] = float("nan")

        myplot = ax.imshow(image,origin="lower",extent=(ras2.min(),ras2.max(),decs2.min(),decs2.max()),aspect=1/np.cos(np.deg2rad(dec_irdc)))
        mycb = fig.colorbar(myplot,ax=ax,fraction=0.046, pad=0.04)

        ax.plot(best_ra,best_dec,"rx",ms=10)

        mycb.set_label("Contamination fraction")

        ax.set_xlim(ras2.min(),ras2.max())
        ax.set_ylim(decs2.min(),decs2.max())

        #ax.plot(culled_cat2["ra"][culled_cat2["phot_g_mean_mag"]<mag_lim],culled_cat2["dec"][culled_cat2["phot_g_mean_mag"]<mag_lim],"w.",ms=2)
        ax.axhline(dec_irdc,color="r")
        ax.axvline(ra_irdc,color="r")

        #ax2.axvline(ra_irdc-irdc_r/np.cos(np.deg2rad(dec_irdc)),color="r")
        #ax2.axvline(ra_irdc+irdc_r/np.cos(np.deg2rad(dec_irdc)),color="r")

        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")

        my_ellipse = Ellipse(xy=(ra_irdc,dec_irdc),height = 2*irdc_r,width=2*irdc_r/np.cos(np.deg2rad(dec_irdc)),facecolor="None",edgecolor="r")
        ax.add_patch(my_ellipse)

        ax.text(ra_irdc+0.1/np.cos(np.deg2rad(dec_irdc)),dec_irdc-0.1,"Cloud Radius",ha="center",va="center",color="r")
    outfile.close()
    fig.tight_layout()

    fig.savefig("irdc_results2/{:09d}_pointing_images.png".format(irdc_identifier),dpi=200,bbox_inches="tight",facecolor='white', transparent=False)