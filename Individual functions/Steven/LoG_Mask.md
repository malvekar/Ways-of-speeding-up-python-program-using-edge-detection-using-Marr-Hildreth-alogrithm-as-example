Work Assigned:
* Creating a function to implement Laplacian of Gaussian also called, Marr-Hildreth Algorithm for Edge Detection.


## **Theory**
---
The Marr-Hildreth or Laplacian of Gaussian (LOG) filter is a discrete Laplacian filter with
particular applications in edge detection. The LOG filter proposed by David Marr and Ellen
Catherine Hildreth tackles the disadvantage of noise sensitivity shown by the Laplacian
operator during edge detection. Thus, the Gaussian undoes high-frequency noise before the
Laplacian detects edges.

The detection criteria are the presence of the zero crossing in the 2nd derivative, combined
with a corresponding large peak in the 1st derivative. Here, only the zero crossings whose
corresponding 1st derivative is above a specified threshold are considered.

The plot of the LOG filter gives a form resembling the traditional Mexican sombrero (hat),
thus giving it the alternate name, Mexican Hat Filter or the Sombrero Filter.

The LOG filter is, as its name suggests, a Gaussian function that has been subjected to the
Laplacian operator. Once convolved with this filter, an image returns in high contrast with
pronounced edges over a zero-intensity ground.

---

#### Mathematical Expression
---

$$LoG(x,y) = \frac{1}{2\pi \sigma^6} [(x^2+y^2)-2\sigma^2]e^-(\frac{x^2+y^2}{2\sigma^2}) $$

---

#### Plots
---
