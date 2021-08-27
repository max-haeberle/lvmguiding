#Library imports
import numpy as np
import matplotlib.pyplot as plt
import time
from astropy.io import fits
from astropy.table import Table,vstack
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astroquery.gaia import Gaia
from scipy.ndimage import gaussian_filter
from scipy import optimize
import healpy as hp
from astropy.table import Table, hstack, vstack
#Instrument specs

class InstrumentParameters:
    def __init__(self,standard_config=True):
        if standard_config:
            self.r_outer=44.5/2         #mm, outer radius (constrained by IFU/chip separation, defines the telescope FoV)
                                 # taken from SDSS-V_0129_LVMi_PDR.pdf Table 7
            self.image_scale=8.92     #1arcsec in microns
                                 # taken from SDSS-V_0129_LVMi_PDR.pdf Table 7
            self.pix_scale=1.01       #arcsec per pixel
                                 # taken from SDSS-V_0129_LVMi_PDR.pdf Table 13
                
            self.a_telescope = np.pi*(16.2/2)**2
                
            self.chip_height=10.2     #mm, guide chip height
                                 # taken from SDSS-V_0129_LVMi_PDR.pdf Table 13
            self.chip_width=14.4      #mm, guide chip width
                                 # taken from SDSS-V_0129_LVMi_PDR.pdf Table 13

            self.pixel_height=1100
            self.pixel_width = 1600
            
            self.bias = 100
            self.gain = 5
            self.dark_current = 15
            self.readout_noise = 5
            
            self.inner_search_radius= 0 #degrees
            self.outer_search_radius= 1000*self.r_outer/self.image_scale /3600

            # guiding limits
            self.mag_lim_lower=17   # mag 
                                 #16.44 mag would be the limit, taken from LVM-0059 Table 8 (optimistic)
            self.mag_lim_upper=0.     # mag, currently no limit
            self.min_neighbour_distance=-1.      #arcsec,

    
            
            self.flux_of_vega = self.a_telescope * 1.6e6 #e-/sec/cm2 #This is for the optimistic case, in the pessimistic case the number would be 8.71e5
            #self.flux_of_vega = a_telescope * 8.71e5
            self.zp = -2.5*np.log10(self.flux_of_vega)

            self.IFU_x = None
            self.IFU_y = None
            self.IFU_r = None
            ## the position of the guide chips in each telescope
            #self.sciIFU_pa1=90        #degrees, E of N
            #self.sciIFU_pa2=270       #degrees, E of N
            #self.skyCal_pa1=90        #degrees, E of N
            #self.skyCal_pa2=270       #degrees, E of N
            #self.spectrophotometric_pa=0 #degrees, E of N, assume fixed (don't account for lack of derotation)

    def load_IFU(self,filename="ifu_positions.xyr"):
        data = np.loadtxt(filename,usecols=(0,1,2))
        self.IFU_x = data[:,0]
        self.IFU_y = data[:,1]
        self.IFU_z = data[:,2]

lvminst = InstrumentParameters(standard_config=True)

#Reading the Gaia catalog
hdul=fits.open("./KK_stars_0-17.fits") #down to 17th mag in G band
cat_full = hdul[1].data # the first extension is a table    
                   # cat['ra'],cat['dec'],cat['phot_g_mean_mag']
hdul.close()



def get_cat_using_healpix(c,plotflag=False,inst=lvminst):
    vec = hp.ang2vec(np.deg2rad(c.dec.value+90),np.deg2rad(c.ra.value))

    ipix_disc = hp.query_disc(nside=64, vec=vec, radius=np.deg2rad(inst.outer_search_radius),inclusive=True)
    print(ipix_disc)

    if plotflag:
        fig,ax = plt.subplots(figsize=(12,12))
        ax.set_aspect("equal")
        ax.axhline(c.dec.value)
        ax.axvline(c.ra.value)
    counter=0
    for ipix in ipix_disc:
        filename = "Gaia_Healpix_64/{:06d}.fits".format(ipix)

        hdul = fits.open(filename)
        data= Table(hdul[1].data)
        print(filename,len(data))
        #data = data.filled()
        if plotflag:
            ax.plot(data["ra"],data["dec"],".")
        if counter==0:
            data_combined = data
            counter+=1
        else:
            data_combined = vstack([data_combined, data])
    return data_combined


def get_cat_using_healpix2(c,plotflag=False,inst=lvminst,verbose=False):
    vec = hp.ang2vec(np.deg2rad(-c.dec.value+90),np.deg2rad(c.ra.value))

    ipix_disc = hp.query_disc(nside=64, vec=vec, radius=np.deg2rad(inst.outer_search_radius),inclusive=True,nest=True)
    if verbose: print(ipix_disc)

    if plotflag:
        fig,ax = plt.subplots(figsize=(12,12))
        ax.set_aspect("equal")
        ax.axhline(c.dec.value)
        ax.axvline(c.ra.value)
    counter=0
    for ipix in ipix_disc:
        filename = "Gaia_Healpix_6/lvl6_{:06d}.npy".format(ipix)
        data = np.load(filename)
        #hdul = fits.open(filename)
        #data= Table(hdul[1].data)
        if verbose: print(filename,len(data))
        #data = data.filled()
        if plotflag:
            ax.plot(data["ra"],data["dec"],".")
        if counter==0:
            data_combined = data
            counter+=1
        else:
            #data_combined = vstack([data_combined, data])
            data_combined = np.concatenate([data_combined, data])
    return data_combined




def in_box(xxs,yys,pa_deg,inst=lvminst):
    # tells you if an xy coordinate (in the focal plane) is on the guider chip
    # based on the instrument specs (above)
    # xy coordinates follow the definitions laid out in LVM-0040_LVM_Coordinates.pdf
    #
    #input:
    #xxs and yys:  np array with x,y positions (in mm)
    #pa:   position angle (east of north) in degrees
    #
    #returns: np array set to 1 (in box) or 0 (not in box)
    #        x&y positions on the chip
    #        note! These do not account for the 6th mirror, which flips the handedness

    #convert position angle to radians
    pa=pa_deg/180.*np.pi
    
    #find some vertices, A and B, of the box (see Photo_on13.07.20at13.58.jpg)
    Ar=inst.r_outer
    Atheta=np.arcsin(inst.chip_width/2./inst.r_outer)
    
    phi=(np.pi-Atheta)/2.
    h1=inst.chip_width/(2.*np.tan(phi))
    h2=inst.r_outer-inst.chip_height-h1
    
    chi=np.arctan(h2/(inst.chip_width/2.))
    
    Br=np.sqrt(inst.chip_width*inst.chip_width/2./2.+h2*h2)
    Btheta=np.pi/2.-chi
            
    
    #convert from polar to cartesian
    Ay=Ar*np.cos(Atheta)
    Ax=Ar*np.sin(Atheta)
    
    By=Br*np.cos(Btheta)
    Bx=Br*np.sin(Btheta)
    
    #print(Ax,Ay,Bx,By)
    
    #are the positions to test within the guider chip?
    #derotate in pa
    rrs=np.sqrt(xxs*xxs+yys*yys)
    thetas=np.arctan2(yys,xxs)
    
    derot_thetas=thetas-pa
    
    derot_xxs=rrs*np.cos(derot_thetas)
    derot_yys=rrs*np.sin(derot_thetas)
        
        
    #compare with box edges
    flagg=np.array(len(xxs)*[False])
    
    ii=((derot_xxs < Ax) & (derot_xxs > -1.*Ax) & (derot_yys < Ay) & (derot_yys > By))
    flagg[ii]=True
        
    #return flags testing whether object is on the chip
    #also return x,y position on chip (in mm), in chip coordinates
    #origin is arbitrarily at (Bx,By) the lower left corner
    #note! this does not account for the 6th mirror, which flips the handedness
    return flagg,derot_xxs+Bx,derot_yys-By

def ad2xy(cats2,c_icrs,inst=lvminst):
    # converts ra/dec positions to angular offsets from field center (c_icrs)
    #
    # inputs:
    # cats2: table of SkyCoords with ra & dec positions 
    # c_icrs: SkyCoord of the IFU field center
    # returns: np array of x and y positions in focal plane, in mm
    
    #convert ra&dec of guide stars to xy position in relation to field center (in arcsec)

    
    # For loop of Kathryn
    #dd_y=np.zeros(len(cats2))
    #dd_x=np.zeros(len(cats2))
    #for i in range(dd_y.size):

    #    ww1=cats2[i]
    #    dd_y[i]=sphdist(ww1['ra'],c_icrs.dec.deg,ww1['ra'],ww1['dec'])*3600. #in arcsec
    #    if ww1['dec'] < c_icrs.dec.deg: 
    #        dd_y[i]*=(-1.)
    #    dd_x[i]=sphdist(c_icrs.ra.deg,ww1['dec'],ww1['ra'],ww1['dec'])*3600. #in arcsec
    #    if ww1['ra'] > c_icrs.ra.deg: 
    #        dd_x[i]*=(-1.)

    #Without the slow loop (Max)
    dd_y = sphdist(cats2['ra'],c_icrs.dec.deg,cats2['ra'],cats2['dec'])*3600. #in arcsec 
    dd_y *= (2*(cats2['dec'] > c_icrs.dec.deg).astype(float)-1)
    
    dd_x=sphdist(c_icrs.ra.deg,cats2['dec'],cats2['ra'],cats2['dec'])*3600. #in arcsec
    dd_x *= (2*(cats2['ra'] < c_icrs.ra.deg).astype(float)-1)  
    
    #convert to mm
    dd_x_mm=dd_x*inst.image_scale/1e3
    dd_y_mm=dd_y*inst.image_scale/1e3
    
    return dd_x_mm,dd_y_mm
    
def deg2rad(degrees):
    return degrees*np.pi/180.

def rad2deg(radians):
    return radians*180./np.pi

def sphdist (ra1, dec1, ra2, dec2):
# measures the spherical distance in degrees
# The input has to be in degrees too
    dec1_r = deg2rad(dec1)
    dec2_r = deg2rad(dec2)
    ra1_r = deg2rad(ra1)
    ra2_r = deg2rad(ra2)
    return 2*rad2deg(np.arcsin(np.sqrt((np.sin((dec1_r - dec2_r) / 2))**2 + np.cos(dec1_r) * np.cos(dec2_r) * (np.sin((deg2rad(ra1 - ra2)) / 2))**2)))
#    return rad2deg(np.arccos(np.sin(dec1_r)*np.sin(dec2_r)+np.cos(dec1_r)*np.cos(dec2_r)*np.cos(np.abs(ra1_r-ra2_r))))
    

def find_guide_stars(c, pa, plotflag=False, remote_catalog=False, east_is_right=True,cull_cat=True,recycled_cat = None,return_focal_plane_coords=False,remote_maglim=None,inst=lvminst):
    # function to figure out which (suitable) guide stars are on the guider chip
    # input:
    # c in SkyCoord;          contains ra & dec of IFU field center
    # pa in degrees;          position angle (east of north) for the guider 
    # plotflag = True;        shows a scatter plot of stars & those selected
    # remote_catalog = True;  queries the GAIA TAP to obtain a catalog on the fly
    # east_is_right = True;   accounts for the handedness fip resulting from the 5 mirror configuration
    #                         should be set to FALSE for the spectrophotometric telescope, which has only 2 mirrors
    #
    # returns: ra,dec, G-band magnitude of all stars on the chip, as well as xy pixel positions (in mm)
                        #note! this does not account for the 6th mirror, which flips the handedness

    inner_search_radius=inst.inner_search_radius
    outer_search_radius=inst.outer_search_radius
    global cat 
    
    #make sure c is in icrs
    c_icrs=c.transform_to('icrs')   
    
    #if not using a local 4GB copy of the catalog, download the appropriate sub-set of gaia data
    if recycled_cat is not None:
        cat = recycled_cat
    else:
        if remote_catalog:
            if remote_maglim is None:
                remote_maglim = inst.mag_lim_lower
            radius = u.Quantity(outer_search_radius, u.deg)
            #j = Gaia.cone_search_async(coordinate=c_icrs, radius)
            gaia_query = "SELECT source_id, ra,dec,phot_g_mean_mag FROM gaiaedr3.gaia_source WHERE phot_g_mean_mag <= "+str(remote_maglim)+" AND 1=CONTAINS(POINT('ICRS',ra,dec), CIRCLE('ICRS',"+str(c_icrs.ra.deg)+","+str(c_icrs.dec.deg)+", "+str(radius.value)+"))"
            print("Gaia query: ",gaia_query)
            j = Gaia.launch_job_async(gaia_query)
            cat = j.get_results()
            #print("Gaia query: ",gaia_query)
            print(f'{len(cat)} stars found within {radius}')

        elif cull_cat:
            t0 = time.time()
            #print("Culling the catalog: ")
            cat_ra = cat_full[(c_icrs.ra.deg - 2/np.cos(c_icrs.dec.rad) < cat_full['ra'])& (cat_full['ra'] < c_icrs.ra.deg +2/np.cos(c_icrs.dec.rad))]
            #print(len(cat_ra)," of ",len(cat_full),"pass initial RA selection")
            cat = cat_ra[(c_icrs.dec.deg - 2 < cat_ra['dec'])& (cat_ra['dec'] < c_icrs.dec.deg +2)]
            #print(len(cat)," of ",len(cat_ra),"pass initial DEC selection")
            t1 = time.time()
            #print("Culling complete. It took me {:.1f} s".format(t1-t0))
        
    #print("Circular selection")
    t0 = time.time()
    dd=sphdist(c_icrs.ra.deg,c_icrs.dec.deg,cat['ra'],cat['dec'])

    #pick the subset that is within 1.5 degree
    #also check some magnitude range
    
    ii=(dd < outer_search_radius) & (dd > inner_search_radius)  & (cat['phot_g_mean_mag'] < inst.mag_lim_lower) & (cat['phot_g_mean_mag'] > inst.mag_lim_upper)
    cats2=cat[ii]
    t1 = time.time()
    #print("Circular selection complete. It took me {:.1f} s".format(t1-t0))
    #convert ra&dec of guide stars to xy position in relation to field center (in mm)
    
    #print("Scale conversion...")
    t0 = time.time()
    dd_x_mm,dd_y_mm=ad2xy(cats2,c_icrs,inst=inst)
    
    if east_is_right:
        dd_x_mm=-1.*dd_x_mm
        pa=-1.*pa
    t1 = time.time()
    #print("Scale conversion complete. It took me {:.1f} s".format(t1-t0))
    #show some star positions in focal plane units
    if plotflag:
        fig, ax1 = plt.subplots(figsize=(12,12))
        #fig = plt.figure(figsize=(6,6)) # default is (8,6)
        ax1.plot(dd_x_mm,dd_y_mm,"k.",ms=2)
        ax1.axis('equal')
        ax1.set_xlabel('x-offset [mm]')
        ax1.set_ylabel('y-offset [mm]')
        ax1.set_ylim(min(dd_y_mm),max(dd_y_mm))
        ax1.set_xlim(min(dd_x_mm),max(dd_x_mm))
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('dec [deg]')
        ax2.set_ylim(min(cats2['dec']),max(cats2['dec']))
        ax3 = ax2.twiny()
        ax3.set_xlabel('ra [deg]')
        if east_is_right:
            ax3.set_xlim(min(cats2['ra']),max(cats2['ra']))
        else:
            ax3.set_xlim(max(cats2['ra']),min(cats2['ra']))

    if return_focal_plane_coords:
        return dd_x_mm,dd_y_mm,cats2
    #identify which stars fall on the guider chip
    selection_on_chip,chip_xxs,chip_yys=in_box(dd_x_mm,dd_y_mm,pa,inst=inst)
    
    iii=np.equal(selection_on_chip,1)
    
    #overplot the identified guide stars positions
    if plotflag:
        iii=np.equal(selection_on_chip,1)
        #print('number of stars on the guide chip: ',np.sum(iii))
        
    
    #also check they aren't crowded. no neighbor within guide_widow (set in preamble)
    
    #print("Before crowding check: ",np.sum(iii),len(iii))
    
    selection_not_crowded = (selection_on_chip==True)
    if inst.min_neighbour_distance>0:
        ctr=0
        for cindex,cc in enumerate(cats2):
            if selection_on_chip[cindex]:
                #dd=cc.separation(cats2[iii])
                dd=sphdist(cc['ra'],cc['dec'],cats2['ra'],cats2['dec'])*3600. #in arcsec
                if np.any((dd > 0) & (dd < inst.min_neighbour_distance)):
                    selection_not_crowded[cindex]=False
                    ctr+=1
                    #print(cindex,"Sum of not crowded:",np.sum(selection_not_crowded))
        #print("Flags: ",np.sum(selection_not_crowded))
        #print(selection_on_chip.dtype,selection_not_crowded.dtype)
        iii=selection_on_chip & selection_not_crowded
        print("Removed {} stars due to crowding.".format(ctr))
    
    #print("After crowding check: ",np.sum(iii),len(iii))
    #overplot the final cleaned guide star positions
    if plotflag:
        #iii=np.equal(flags,1)
        #print('after checking for crowding: ',np.sum(iii))
        ax1.plot(dd_x_mm[iii],dd_y_mm[iii],"c.",ms=4)
        #ax1.plot(dd_x_mm[iii][mags<12],dd_y_mm[iii][mags<12],"b.",ms=6)
        plt.show()
    
    #sub select the guide stars that fall on the guide chip, return their properties
    ras=cats2['ra'][iii]
    decs=cats2['dec'][iii]    
    mags=cats2['phot_g_mean_mag'][iii]
    #print("iii Selection: {} of {}".format(np.sum(iii),len(iii)))
    chip_xxs = chip_xxs[iii]
    chip_yys = chip_yys[iii]
    
    dd_x_mm = dd_x_mm[iii]
    dd_y_mm = dd_y_mm[iii]

        
    return ras,decs,dd_x_mm,dd_y_mm,chip_xxs,chip_yys,mags,cats2
    

def find_guide_stars_auto(input_touple,inst=lvminst,folder="/data/beegfs/astro-storage/groups/others/neumayer/haeberle/lvm_outsourced/guide_star_search_results_no_faint_limit/",verbose=False,save_bin = True):
    #print(inst.mag_lim_lower)
    t0 = time.time()
    index = input_touple[0]
    c = input_touple[1]
    if verbose: print("Analyzing pointing {} (ra: {} dec: {})\n".format(index+1,c.ra.deg,c.dec.deg))
    #c = SkyCoord(frame="galactic", l=280, b=0,unit='deg')
    #print(c)
    for pa in [0,60,120,180,240,300]:
        #qqprint("PA: ",pa)
        if pa==0:
            culled_cat=get_cat_using_healpix2(c,plotflag=False,inst=inst)
        ras,decs,dd_x_mm,dd_y_mm,chip_xxs,chip_yys,mags,culled_cat = find_guide_stars(c,pa=pa,plotflag=False,recycled_cat=culled_cat,inst=inst)    

        output = np.stack((ras,decs,dd_x_mm,dd_y_mm,chip_xxs,chip_yys,mags),axis=1)
        
        if save_bin:
            filename = folder+"guide_stars_{:06d}_pa_{:03d}.npy".format(index,pa)
            np.save(filename,output)
        else:
            filename = folder+"guide_stars_{:06d}_pa_{:03d}".format(index,pa)
            np.savetxt(filename,output,fmt="%10.6f")
        
    return index,len(ras),time.time()-t0




def make_synthetic_image(chip_x,chip_y,gmag,inst,exp_time=5,seeing_arcsec=3.5, sky_flux=10,plotflag = True,write_output=None):
    seeing_pixel = seeing_arcsec*inst.image_scale / (inst.chip_width/inst.pixel_width*1000) / 2.36
    
    x_position = chip_x / inst.chip_width * inst.pixel_width
    y_position = chip_y / inst.chip_height * inst.pixel_height
    
    selection_on_chip = (0<x_position) & (x_position < inst.pixel_width) & (0<y_position) & (y_position < inst.pixel_height)
    
    x_position = x_position[selection_on_chip]
    y_position = y_position[selection_on_chip]
    gmag = gmag[selection_on_chip]
    
    print("{} of {} stars are on the chip.".format(np.sum(selection_on_chip),len(selection_on_chip)))
    
    #gaia_legend_mag = np.arange(17,4,mag_lim_index)
    #gaia_legend_flux= 10**(-(np.array(gaia_legend_mag)+zp)/2.5)
    
    gaia_flux = 10**(-(gmag+inst.zp)/2.5)

    background = (sky_flux+inst.dark_current)*exp_time
    n_pix = 7*7


    background_noise = np.sqrt(background+inst.readout_noise**2)
    signal = gaia_flux*exp_time
    noise = np.sqrt(inst.readout_noise**2+signal+n_pix*background)


    sn = signal/noise


    
    star_image = np.zeros((inst.pixel_height,inst.pixel_width))

    for index, current_flux in enumerate(gaia_flux):
        current_x = x_position[index]
        current_y = y_position[index]

        i = int(current_x)
        j = int(current_y)

        xx = current_x-i
        yy = current_y-j

        #
        #test_image[j,i] = current_f


        star_image[j,i] = (1-xx)*(1-yy)*current_flux*exp_time


        if i<inst.pixel_width-1:
            star_image[j,i+1] = (xx)*(1-yy)*current_flux*exp_time
        if j< inst.pixel_height-1:
            star_image[j+1,i] = (1-xx)*(yy)*current_flux*exp_time
        if (i<inst.pixel_width-1) & (j<inst.pixel_height-1):
            star_image[j+1,i+1] = (xx)*(yy)*current_flux*exp_time

        #star_image[int(current_y),int(current_x)] = current_flux*exp_time

    star_image_c = gaussian_filter(star_image, sigma=seeing_pixel,mode="constant")
    star_image_c_noise = np.random.poisson(lam=star_image_c,size = star_image_c.shape)

    
    background_array = np.random.poisson(background,size=star_image.shape)

    readout_noise_array = np.random.normal(loc=0,scale = inst.readout_noise,size=star_image.shape)

    combined = star_image_c_noise + background_array + readout_noise_array + inst.bias

#hdu = fits.PrimaryHDU(star_image_c)
#hdu.writeto("/home/haeberle/exchange/lvm/synthetic_image_sparse_field_5s.fits",overwrite=True)


    
    if write_output is not None:
        
        hdu = fits.PrimaryHDU(combined)


        #filename = "/home/haeberle/exchange/lvm/synthetic_images/pointing_"+pointing_string+"_{:d}ms.fits".format(int(1000*exp_time))
        print("writing file: ",write_output)
        hdu.writeto(write_output,overwrite=False)

    print("Nstars: ",x_position.shape)
    
    return combined

def calc_sn(gmag,inst,n_pix=7*7,sky_flux=10,exp_time=5):
        gaia_flux = 10**(-(gmag+inst.zp)/2.5)

        background = (sky_flux+inst.dark_current)*exp_time


        background_noise = np.sqrt(background+inst.readout_noise**2)
        signal = gaia_flux*exp_time
        noise = np.sqrt(inst.readout_noise**2+signal+n_pix*background)


        sn = signal/noise

        return sn