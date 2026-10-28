# VISTA provenance and upstream notices

Algorithm: Rizwan Ahmad, Hui Xue, Shivraman Giri, Yu Ding, Jason Craft,
and Orlando P. Simonetti, "Variable Density Incoherent Spatiotemporal
Acquisition (VISTA) for Highly Accelerated Cardiac MRI," *Magnetic Resonance
in Medicine* 74:1266-1278 (2015), DOI 10.1002/mrm.25507.

Executable reference: OSU-CMR/VISTA, <https://github.com/osu-cmr/vista>, pinned
to commit `1ad7ea0130247aca220a3e761fc8a64b770900b2`. The Python implementation
in this directory is a CMRBench adaptation/port.

The following notice is reproduced from the OSU-CMR/VISTA repository:

> Copyright (c) 2014 - The Ohio State University.
> All rights reserved.
>
> Permission to use, copy, modify, and distribute this software and its
> documentation for educational, research, and not-for-profit purposes,
> without fee and without written agreement, is hereby granted, provided that
> the above copyright notice, the following two paragraphs, and the author
> attribution appear in all copies of this software. For commercial licensing
> possibilities, contact The Office of Technology Commercialization Office
> (http://tco.osu.edu/) at The Ohio State University.
>
> IN NO EVENT SHALL THE OHIO STATE UNIVERSITY BE LIABLE TO ANY PARTY FOR
> DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
> OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE OHIO STATE
> UNIVERSITY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> THE OHIO STATE UNIVERSITY SPECIFICALLY DISCLAIMS ANY WARRANTIES INCLUDING,
> BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
> FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
> BASIS, AND THE OHIO STATE UNIVERSITY HAS NO OBLIGATION TO PROVIDE
> MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

The upstream `randp.m` is attributed there to Jos van der Geest (version 2.0,
February 2009) and carries its own warranty disclaimer. This port does not copy
that implementation; `_weighted_random_indices` independently implements the
documented operation (weighted discrete draws with replacement) using NumPy.
