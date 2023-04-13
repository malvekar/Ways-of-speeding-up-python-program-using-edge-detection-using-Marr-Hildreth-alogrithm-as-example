#### Work Assigned:
* Creating a function to implement Laplacian of Gaussian also called, Marr-Hildreth Algorithm for Edge Detection.


---
---
### **Theory**

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


$$LoG(x,y) = \frac{1}{2\pi \sigma^6} [(x^2+y^2)-2\sigma^2]e^-(\frac{x^2+y^2}{2\sigma^2}) $$

---

#### Plot

<p align="center">
  <img width="410" height="332" src="https://github.com/malvekar/Ways-of-speeding-up-python-program-using-edge-detection-using-Marr-Hildreth-alogrithm-as-example/blob/main/Individual%20functions/Steven/marr_hildreth.png">
</p>

---

[!faq]- Code:

#### Parameters

- Size: Specifies the size of the mask to be implemented.
- Sigma: Specifies the standard deviation of the Gaussian.
- Const: This is a constant which is multiplied by the LOG.

---

#### Working

The function takes in the parameters specified by the user, creates a mask of the specified
size. Each position in the matrix is iterated and appended with the Laplacian of the Gaussian
of the specified sigma by the user. To discretise the mask, it is multiplied with a constant
and then rounded off to the closest integer.

The sum of the elements in the mask is calculated and displayed.

---
---
