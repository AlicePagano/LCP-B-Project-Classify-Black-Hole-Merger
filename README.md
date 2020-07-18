# LCP-Project-Classify-Black-Hole-Mergers

### Abstract

Recently, Ligo and Virgo detections has proved the existence of Binary Black Holes systems. Therefore, it is fundamental to investigate what are the formation channels of these compact objects binaries. Several physical processes, such as mass transfer and Common Envelope, and stars composition parameters, such as metallicity, can affect the formation of Binary Black Hole systems.
In order to highlight relevant trends and correlations in Binary Black Holes formation, we analyze data simulated with the population synthesis code MOBSE focusing on the parameters of our interest.  

### Folder Organization

#### Code.

* 01_merger.

We read the file “mergers.out”, putting together all chunks but keeping different metallicities Z and different mass transfer efficiencies fMT separated.
In particular, we calculate and plot distribution of masses, mass ratios and merging times  for different Z and different fMT. We compute also the fraction of flipped masses as a function of fMT.

* 02_evol-merger.

We read the file “evol_mergers.out” and classify systems between systems those that go through Common Envelope and those that do not.
We calculate and plot distribution of initial and final masses for different metallicities, different fMT and for systems which enter or not in Common Envelope.
Moreover, we compute the fraction of CE systems as a function of fMT.

* 02_1_evol-merger.

We read the file “evol_mergers.out” and classify again systems between systems those that go through Common Envelope and those that do not. We plot the distribution of merging time, mass ratios and semi-major axis as a function of metallicity and fMT. Moreover, we make a scatter plot of $m_1$ vs $m_2$ for initial and final masses.

#### Plots.

* 01_merger.

This folder contains the plots obtained by running the code in folder 01_merger.

* 02_evol-merger.

This folder contains the plots obtained by running the code in folder 02_evol-merger.

* 02_1_evol-merger.

This folder contains the plots obtained by running the code in folder 02_1_evol-merger.


#### Report.
This folder contains the final report with its Latex Source.
