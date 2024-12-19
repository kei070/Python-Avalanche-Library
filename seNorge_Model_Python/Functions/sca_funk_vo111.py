"""
This presents an attempt to implement the helper function sca_funk_vo111 in Python.

The model R-code was downloaded from https://ars.els-cdn.com/content/image/1-s2.0-S0022169416301755-mmc1.zip

The documentation is copied from the original R-script.
"""

# imports
import numpy as np

# function
def sca_funk_vo111(M, Ps, SWEi_old, SWEi_HI_old, SWEi_buff_old,  CF):

    """
    A simple snow-covered area (SCA) algorithm, based on the VIC model
    fourth, final version (checked & documented by Tuomo Saloranta, NVE)
    by Tuomo Saloranta, Norwegian Water Resources and Energy Directorate (NVE), 27.11.12

    === Input ===
    sca_funk_vo111(M, Ps, SWEi_old, SWEi_HI_old, SWEi_buff_old,  CF)

    Parameters:
        M:             melt (negative) /refreeze (positive) rate at current time step [mm, water equivalent]
                       (NB! refreezing must be restricted to actual liquid water availability)
        Ps:            new snow fall  at current time step (SWE, [mm, water equivalent])
        SWEi_old:      ice content in snow [mm, water equivalent] from previous time step
        SWEi_HI_old:   maximum-SWEi-so-far from previous time step [mm, water equivalent]
        SWEi_buff_old: newsnow buffer from previous time step [mm, water equivalent]
        CF:            the variability factor parameter (+- % around the SWEi_HI; e.g. CF=0.3 -> +- 30%)

    === Output ===
    Output.vec = np.array([SWEi_HI, SWEi_buff, SCA, SWEi, fM])

        SWEi_HI:       as SWEi_HI_old, but updated value for current time step
        SWEi_buff:     as SWEi_buff_old, but updated value for current time step
        SCA:           total SCA
        SWEi:          total SWEi (mean over the grid cell, adjusted by the SCA)
        fM:            reduction factor to adjust melting or refreezing (due to SCA < 1)
    """

    # print(f"M = {M}")


    ns_thresh = 1 # minimum threshold [mm, water equivalent] for new snow fall to cover the whole pixel (i.e. SCA=1)

    # === Preparing variables  ===
    # SWEi in the main snow pack (excl. new snow buffer; from previous time step)
    SWEi_pack_old = np.max([0, SWEi_old - SWEi_buff_old])
    # +- variability range [mm, water equivalent] (from previous time step)
    R_SWEi_old = SWEi_HI_old*CF

    if (SWEi_pack_old < R_SWEi_old):  # i.e. SCA_pack_old < 1
        # untruncated "imaginary" mean SWEi (" SWEi_pack* ") from previous time step (excl. new snow buffer, can be
        # negative) [mm, water equivalent]
        SWEi_imag_old = 2*np.sqrt(SWEi_pack_old*R_SWEi_old) - R_SWEi_old
    else:
        SWEi_imag_old = SWEi_pack_old
    # end if else

    # the highest SWEi in the uniform distribution (from previous time step)
    SWEi_max_old = SWEi_imag_old + R_SWEi_old

    # === Copy refreezing to own variable, so that always is M <=0
    refrz = 0
    # if refreezing
    if (M > 0):
        refrz = M
        M = 0
    # end if

    # === Calculate SCA_pack for previous time step (i.e. SCA for the main snow pack (excl. new snow buffer))
    if (SWEi_HI_old == 0):  # implying no snow
        SCA_pack_old = 0
    else:
        SCA_pack_old = np.max([0, np.min([1, SWEi_imag_old/(2*R_SWEi_old) + 0.5])])
    # end if else


    # === Updating variables for the current time step (NB: M is now restricted to values <= 0)

    # === Option 1): if NOT whole SWEi_buff_old is melted
    if ((SWEi_buff_old + M) > 0):

        # print("Option 1: if NOT whole SWEi_buff_old is melted")

        SWEi_buff = SWEi_buff_old + M + Ps + refrz # melt/refreezing + accumulation
        SWEi_pack = SWEi_pack_old
        SCA_pack = SCA_pack_old
        SWEi_max = SWEi_max_old
        SWEi_imag = SWEi_imag_old
        fM = 1

    # === Option 2): if whole SWEi_buff_old is melted, or SWEi_buff_old = 0
    else:
        # print("Option 2: if whole SWEi_buff_old is melted, or SWEi_buff_old = 0")

        # === Alternative 1:
        # implying that SCA < 1 either i) before (to calculate refreezing, or no melt case) or ii) after melting
        if ( (SWEi_old < R_SWEi_old) | ((SWEi_old + M) < R_SWEi_old) ):

            # print("Alternative 1 implying SCA < 1")

            # Here "melting pushes the 'imaginary' snow column down, refreezing rises it up"
            SWEi_imag = SWEi_imag_old + SWEi_buff_old + M + refrz

            SWEi_max = SWEi_imag + R_SWEi_old

            # Check if the whole pack was melted, and calculate remaining melt "energy"
            if (SWEi_max < 0):
                rem_M = SWEi_max #remaining melt "energy" [mm, water equivalent]
                SWEi_imag = 0
                SWEi_max = 0
                SWEi_HI_old = 0
            else:
                rem_M = 0
            # end if else

            # Calculate SCA in the main snow pack after melting (new snow buffer = 0 at this stage)
            if (SWEi_HI_old == 0):  # implying no snow
                SCA_pack = 0
            else:
                SCA_pack =  np.max([0, np.min([1, SWEi_imag/(2*R_SWEi_old) + 0.5])])
            # end if else

            # Update SWEi_pack after melting/refreezing (in three different SCA-cases)
            # == SCA-case 1 (partly snow-covered pixel)
            if ( (SCA_pack > 0) & (SCA_pack < 1) ):
                # print("... 0 < SCA < 1")
                SWEi_pack = SCA_pack*0.5*SWEi_max
                if ( (M == 0) & (refrz == 0) ):
                    fM = 1
                else:
                    # (new snow buffer = 0 at this stage; either M or refrz is zero)
                    fM = (SWEi_pack - SWEi_old)/(M + refrz)
                # end if else
                SWEi_buff = Ps # Update buffer with new snow accumulation
            # end if

            # == SCA-case 2 (exceptional case, SCA goes from <1 to ==1 due to refreezing)
            if (SCA_pack == 1):
                # print("... SCA = 1")

                SWEi_pack = SWEi_imag + Ps
                fM = (SWEi_imag - SWEi_old)/(M + refrz) # (new snow buffer = 0 at this stage; either M or refrz is zero)
                SWEi_buff = 0
            # end if

            # == SCA-case 3 (bare-ground pixel after melting (incl. the case of no snow pack at the start of the time
            # step)
            if (SCA_pack == 0):
                # print("... SCA = 0")

                SWEi_pack =  np.max([0, Ps + rem_M])
                # (new snow buffer = 0 at this stage; either M or refrz is zero). fM limits potential M here to
                # available ice
                fM = (rem_M - SWEi_old)/(M + refrz)
                SWEi_buff = 0
                if ((Ps + rem_M)>0):
                    SCA_pack = 1
                # end if
            # end if

        # === Alternative 2:
        else:  # implying that SCA = 1 after melting  (new snow buffer = 0 at this stage)
            # print("Alternative 2 implying SCA = 1 after melting")
            SCA_pack = 1
            SWEi_pack = SWEi_old + M + refrz + Ps
            fM = 1 # no melt/refreezing reduction
            SWEi_buff = 0 # no buffer
        # END if else (Alternative 1&2)

    # END if else (Option 1&2)

    # === Check if the new snow buffer is full
    if (SWEi_buff > 0):
        if ( SWEi_buff > (SWEi_max/(1+CF)-SWEi_pack) ): # if the buffer is full
            SWEi_pack = SWEi_pack + SWEi_buff
            SWEi_buff = 0
            SCA_pack = 1
            SWEi_HI_old = SWEi_pack
        # end if
    # end if

    # === Update the maximum-SWEi-so-far
    if (SWEi_pack > SWEi_HI_old):
        SWEi_HI = SWEi_pack
    else:
        SWEi_HI = SWEi_HI_old # no change
    # end if

    # === Prepare output
    SWEi = SWEi_pack + SWEi_buff    # Total SWEi
    SCA = SCA_pack                  # Total SCA

    if (SWEi_buff >= ns_thresh):
        # Total SCA = 1 in case of new snow (over treshold limit)
        SCA = 1
    # end if
    if (SWEi < ns_thresh):
        # Total SCA = 0 in case of for very little SWEi (reducing "SCA-noise" in the early snow season)
        SCA = 0
    # end if

    Output_vec = np.array([SWEi_HI, SWEi_buff, SCA, SWEi, fM])

    return Output_vec
# END (function)

