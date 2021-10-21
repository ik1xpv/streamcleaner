<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/PabloMSanAla/fabada">
    <img src="fabada-logo.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Fully Adaptive Bayesian Algorithm for Data Analysis.  FABADA</h3>
<h3 align="center">FABADA</h3>

  <p align="center">
    FABADA is a novel non-parametric noise reduction technique which arise from the point of view of Bayesian inference that iteratively evaluates possible smoothed models of the data, obtaining an estimation of the underlying signal that is statistically compatible with the noisy measurements.
    Iterations stop based on the evidence $E$ and the $\chi^2$ statistic of the last smooth model, and we compute the expected value of the signal as a weighted average of the smooth models.
    You can find the entire paper describing the new procces in (introduce link to arxiv).
    <br />
    <a href="https://github.com/PabloMSanAla/fabada"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/PabloMSanAla/fabada">View Demo</a>
    ·
    <a href="https://github.com/PabloMSanAla/fabada/issues">Report Bug</a>
    ·
    <a href="https://github.com/PabloMSanAla/fabada/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Method</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#usage">Results Paper</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#usage">Cite</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <!-- <li><a href="#acknowledgments">Acknowledgments</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Method

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

This automatic method is focused in astronomical data, such as images (2D) or spectra (1D). Although, this doesn't mean it can be treat like a general noise reduction algorithm and can be use in any kind of two and one-dimensional data reproducing reliable  results. 
The only requisite of the input data is an stimation of its variance.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

We try to make as simple as possible the usage of FABADA. For that purpose we have create a pypi package to install FABADA in its latest version. 

### Prerequisites

The first requirement is to have a version of Python greater than 3.5.
Although pypi install the prerequisites itself, FABADA has two dependecies. 

* Numpy <br />
In order to install using pip
  ```sh
  pip install numpy
  ```
or using conda instead
  ```sh
  conda install numpy
  ```
* Scipy <br />
Same way as numpy
  ```sh
  pip install scipy
  ```
or using conda instead
  ```sh
  conda install -c anaconda scipy
  ```
However both of the packages usually comes installed with python. 

### Installation

Using pip
  ```sh
  pip install fabada
  ```
or using conda instead
  ```sh
  conda install fabada
  ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Along with the package two examples are given. 

* _fabada_demo_image.py_ 

In here we show how to use fabada for an astronomical grey image (two dimensional)
First of all we have to import our library previously install <br />
    ```python
    from fabada import fabada
    ```
Then we read the [bubble image](https://github.com/PabloMSanAla/fabada/blob/master/examples/bubble.png) borrowed from the [Hubble Space Telescope gallery](https://www.nasa.gov/mission_pages/hubble/multimedia/index.html). In our case we use the [OpenCV](https://pypi.org/project/opencv-python/) library for that. We also add some random Gaussian white noise using [numpy.random](https://numpy.org/doc/1.16/reference/routines.random.html).
    ```python
    # IMPORTING IMAGE
    y = cv2.imread("bubble.png",0)

    # ADDING RANDOM GAUSSIAN NOISE
    np.random.seed(12431)
    sig      = 15             # Standard deviation of noise
    noise    = np.random.normal(0, sig ,y.shape)
    z        = y + noise 
    variance = sig**2
    ```
Once the noisy image is generated we can apply fabada to produce an estimation of the uderlying image, which we only have to call fabada and give it the variance of the noisy image
    ```python
    y_recover = fabada(z,variance)
    ```
And its done :wink:. As easy as one line of code. 
The results obtained running this example would be:

[![Product Name Screen Shot][image_results]]


* _fabada_demo_spectra.py_

In here we show how to use fabada for an astronomical spectra (one dimensional), basically is the same as the example above since fabada is the same for one and two dimensional data. 
First of all we have to import our library previously install 
    ```python
    from fabada import fabada
    ```
Then we read the interacting galaxy pair [Arp 256](http://simbad.u-strasbg.fr/simbad/sim-basic?Ident=arp256&submit=SIMBAD+search) spectra, taken from the [ASTROLIB PYSYNPHOT](https://github.com/spacetelescope/pysynphot) package which is store in [arp256.csv](https://github.com/PabloMSanAla/fabada/blob/master/examples/arp256.csv). Again we  add some random Gaussian white noise 
    ```python
    # IMPORTING SPECTRUM
    y = np.array(pd.read_csv('arp256.csv').flux)[100:1530]
    y = (y/y.max())*255  # Normalize to 255

    # ADDING RANDOM GAUSSIAN NOISE
    np.random.seed(12431)
    sig      = 10             # Standard deviation of noise
    noise    = np.random.normal(0, sig ,y.shape)
    z        = y + noise 
    variance = sig**2
    ```
Once the noisy image is generated we can, again, apply fabada to produce an estimation of the uderlying spectrum, which we only have to call fabada and give it the variance of the noisy image
    ```python
    y_recover = fabada(z,variance)
    ```
And its done :wink:. As easy as one line of code. 
Which is exactly the same as for two dimensional data. 

The results obtained running this example would be:

[![Product Name Screen Shot][image_results]]



_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- Results Paper -->
## Results Paper

All the results of the paper of this algorithm can be found in the  [GitHub Pages](https://pablomsanala.github.io/fabada/) of this repositorie. 

_Note: This page might be under development at this moment_


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GNU General Public License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Pablo M Sanchez Alarcón  - pablom.sanala@gmail.com

Yago Ascasibar Sequerios - yago.ascasibar@uam.es

Project Link: [https://github.com/PabloMSanAla/fabada](https://github.com/PabloMSanAla/fabada)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS 
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p> -->


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/PabloMSanAla/fabada.svg?style=plastic&logo=appveyor
[contributors-url]: https://github.com/PabloMSanAla/fabada/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/PabloMSanAla/fabada.svg?style=for-the-badge
[forks-url]: https://github.com/PabloMSanAla/fabada/network/members
[stars-shield]: https://img.shields.io/github/stars/PabloMSanAla/fabada.svg?style=for-the-badge
[stars-url]: https://github.com/PabloMSanAla/fabada/stargazers
[issues-shield]: https://img.shields.io/github/issues/PabloMSanAla/fabada.svg?style=for-the-badge
[issues-url]: https://github.com/PabloMSanAla/fabada/issues
[license-shield]: https://img.shields.io/github/license/PabloMSanAla/fabada.svg?style=for-the-badge
[license-url]: https://github.com/PabloMSanAla/fabada/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[image_results]: src/images/image_results.png
[spectra_results]: src/images/spectra_results.png

