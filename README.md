# Hello Regression!

Instead of the standard **Hello World!** program, I find it useful to write the Ordinary Least Squares (OLS) algorithm when experimenting with data science focused languages for the first time.

This repo contains examples of implementing Multi Linear Regression using OLS in the following languages:
* R
* Julia
* Python

<br>

## Example Output

* ### R
	```bash
	$ Rscript mlr.r
	Model parameters: 
		-12.57533 0.9792538 [...] -2.221289 

	Out-of-Sample R-Squared: 
		0.3157442
	```

* ### Julia
	```bash
	$ julia mlr.jl
	Model parameters: 
		[-12.57532511691548, 0.9792537885948169, ..., -2.221289297683287]

	Out-of-Sample R-Squared: 
		0.3157441726897712
	```

* ### Python
	```bash
	$ python mlr.py
	Model parameters:
		[[-1.25753251e+01  9.79253789e-01 ... -2.22128930e+00]]

	Out-of-Sample R-Squared: 
		0.3157441726893673
	```

<br>

## Notes

* All of the MLR examples follow the same layout, and because they use the same data to create the linear model, they return the same output
* The `cars.csv` from which the models are built is the built-in `mtcars` dataset that comes with R