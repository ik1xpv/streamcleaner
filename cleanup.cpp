/*       			The Asshole "Free but not really `Open Source`" Hacker's License Agreement
		 By the grace of God, and the merciful love of the Savior, the Lord Jesus Christ, the software is free as in free beer.
	I made the code and I offer it freely to you, and you owe me nothing, but you cannot resell the beer, it was free!
	If you supply the beer to someone else and get paid for the act of supplying the free beer to them, you must negotiate my share of pay with me.
	whiteroom it or attempt by any shenanigans at all to implement my code into DSP hardware and make money for it and I will sue your ass off.

	Copyright 2023 Joshuah Rainstar
1.	The source code provided here (hereafter, "the Software"), is copyrighted by Joshuah Rainstar (hereafter, “Author”)
	and ownership of all right, title and interest in and to the Software remains with Author.

2.	The definition of "Use" shall include copying, redistributing, modifying, compiling, interpreting, translating, making derivatives in other programming
	languages of, making emulations based on measuring the capability of, reading, whiterooming(which means reading it, explaining what it does to someone else,
	even an AI, where they make code to emulate the function which is original) and running or executing anything derived from or based on the Software.

3.	By Using the Software, you (hereafter, "User") agree to abide by the terms of this Agreement, which are as follows:
4.	You may not use the software for any purpose without including this license.
5.	The Agreement will terminate immediately upon User's breach of, or non-compliance with, any of its terms.
	User may be held liable for any copyright infringement or the infringement of any other proprietary rights in the Software.

6.	Author grants to User a royalty-free, nonexclusive right to Use the software for any purpose, subject to the following conditions:

7.	If the User represents a government or legally incorporated business entity, the User agrees that, outside of contractual negotiation with Author,
	the software may only be Used on the machines in one building that the User operates in.

8.	For fire departments, emergency rescue, and paramedic/rescue only operations only, which do not engage in law enforcement or military activities,
	and for direct usage only, where the User is the entity itself and not a contractor offering them services, part 7 of the
	Agreement shall not appply and user is excempt.	If, however, at any time, the system integrating the software is contractually
	sub-delegated at cost to a third party to operate,  the user agrees their exception from the provisions in part 7 of the agreement shall terminate.

9.	Regardless of the location or juresdiction of the User's nation, the User agrees to honor the Agreement, and to be subject to it as if it were agreed to
	for purposes, activities, uses, and contracts within and subject to, construed and enforced in accordance with the law of the state of Colorado.
	The parties irrevocably consent to the exclusive jurisdiction of the state or federal courts located in Denver for all disputes concerning this agreement.

10.	User acknowledges that the Software is supplied "as is," without any absolute guarantee of support services from Author.
	Author makes no representations or warranties, express or implied, including, without limitation,
	any representations or warranties of the merchantability or fitness for any particular purpose,
	or that the application of the software, will not infringe on any patents or other proprietary rights of others.

11.	Author shall not be held liable for direct, indirect, incidental or consequential damages arising from any claim by User
	or any third party with respect to uses allowed under the Agreement, or from any use of the Software.

12.	User agrees to fully indemnify and hold harmless Author from and against any and all claims, demands, suits, losses, damages,
	costs and expenses arising out of the User's use of the Software, including, without limitation, arising out of the User's modification of the Software.

13.	The user agrees that the definition of `modified work` shall include any part of the Software implemented as part of any other program,
	shall include any works made by reverse engineering, emulating, examining, or otherwise whiteroom-implementing the Software,
	including but not limited to the use of Artificial Intelligence, in addition to any modified work based upon the Software.
14.	For the purposes of `modified work` as agreed to in part 10 of the Agreement, User agrees to the following additional conditions:
	User may Use, modify the Software and make `modified work` with regards to the Software and and distribute it to to third parties provided that:
	(a) If posted separately, it clearly acknowledges that it contains material copyrighted by the Author,
	(b) User agrees to notify the Author of the distribution,
	(c) User clearly notifies secondary users that such modified work is not the original Software by including the License Agreement,
	(d) User agrees that any `modified work` must either be donationware, freeware, open source, or not provided at cost.
	(e) User agrees that with regards to `DSP` devices which incorporate a `modified work` based upon the Use of the Software,
	the Author must be paid 10% of the retail cost of every unit ever sold by User.
	(e) User agrees to provide Author with a non-irrevocable non-exclusive interest in ownership of the `modified work`,  which is limited to the following condition:
	(f) User agrees that if they are paid for a contract or work that involves the Use of the Software or a `modified work` derived from the Software,
	or any other work in a chain of Uses of said works which ultimately are derived from any `modified work` which was at one time produced by the Use of the Software,
	the User will provide the Author with 1% of the value of the contract.

15. On January 1st, 2100, The terms of this License shall be considered null and void and the Software, provided the author remains alive,
	which is unlikely, shall be considered dually licensed under the MIT license and the GPL-2.0, in perpetuum.
16.	In the event that the Author both has a suitable lover and helpmate to their own preference, and has also exceeded a
	lifetime collective income after taxes of 2,799,360 USD, Author agrees to relicense the code as dually licensed under the MIT license and the GPL-2.0, in perpetuum.
17.	The author does not give a shit about what you think of the license agreement and User agrees not to complain about it.
*/

//please note: this project is still a work in process and is not finished.
//please do not attempt to use this code for any purpose until this line is removed.


/*
* Cleanup. CPP version 0.01
*  execute time with intelDcc compiler, all things optimized.. 153ms of time to compute 2800ms of audio. This is 18x speedup
*  not sure how much of the code runs though, lol!
*
*
*/

#include <iostream>
#include <vector>
#include <complex>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <array>









/*i have not yet validated that the c++ performs as the python
def man(data: numpy.ndstd::array):
  arr = data
  arr = data[0:NBINS] #change this line so that any NAN are removed from arr
  v = arr[numpy.nonzero(arr)] # if v is empty, return 0.0 early
  t = numpy.median(v)

  # Compute the median of the absolute of arr minus t - this is NOT median absolute deviation
  non_zero_diff = numpy.abs(arr- t)
  e = numpy.median(non_zero_diff)

  # Compute the  square root of the mean of the squared absolute deviation from e
  x = numpy.square(numpy.abs(arr - e))
  a = numpy.sqrt(numpy.mean(x))
return a
*/
//"Median Absolute deviation root mean square with Nonzero median thresholding"
float MAN(std::array<float, 257>& data, int& NBINS) {
	std::array<float, 257> arr;
	int n = 0;

	// Remove NaN values from arr
	for (int i = 0; i < NBINS; i++) {
		if (!isnan(data[i])) {
			arr[n] = data[i];
			n++;
		}
	}

	// If v is empty, return 0.0 early
	if (n == 0) {
		return 0.0;
	}

	// Compute the median of arr
	sort(arr.begin(), arr.begin() + n);
	float t = (n % 2 == 0) ? (arr[n / 2] + arr[(n / 2) - 1]) / 2.0 : arr[n / 2];

	// Compute the median of the absolute difference between arr and t
	std::array<float, 257> non_zero_diff;
	for (int i = 0; i < n; i++) {
		non_zero_diff[i] = abs(arr[i] - t);
	}
	sort(non_zero_diff.begin(), non_zero_diff.begin() + n);
	float e = (n % 2 == 0) ? (non_zero_diff[n / 2] + non_zero_diff[(n / 2) - 1]) / 2.0 : non_zero_diff[n / 2];

	// Compute the square root of the mean of the squared absolute deviation from e
	float sum_x = 0.0;
	for (int i = 0; i < n; i++) {
		float abs_diff = abs(arr[i] - e);
		sum_x += abs_diff * abs_diff;
	}
	float a = sqrt(sum_x / n);

	return a;
}

void threshold(std::array<std::array<float, 257>, 192>& data, int& NBINS, float& threshold) {
	// Compute the median of the absolute values of the non-zero elements
	float median = 0.0;
	int count = 0;
	for (int j = 0; j < NBINS; j++) {
		for (int i = 0; i < 192; i++) {
			float val = data[i][j];
			if (val != 0.0) {
				median += abs(val);
				count++;
			}
		}
	}
	if (count == 0) {
		threshold = 0.0;
		return;
	}
	median /= count;

	// Compute the threshold using the formula from the original implementation
	float sum = 0.0;
	int n = 0;
	for (int j = 0; j < NBINS; j++) {
		for (int i = 0; i < 192; i++) {
			float val = data[i][j];
			if (!isnan(val) && isfinite(val)) {
				float diff = abs(val - median);
				sum += diff * diff;
				n++;
			}
		}
	}
	if (n == 0) {
		threshold = 0.0;
		return;
	}
	threshold = sqrt(sum / n) + median;
}





void same_vector_convolve(std::vector<float>& input1, const std::vector<float>&input2) {
	int size1 = input1.size();
	int size2 = input2.size();
	int size_out = size1;

	// Zero-pad the input vector
	std::vector<float> input1_padded(size1 + size2 - 1, 0.0);
	std::copy(input1.begin(), input1.end(), input1_padded.begin() + (size2 - 1) / 2);

	// Reverse the second input vector
	std::vector<float> reversed_input2(input2.rbegin(), input2.rend());

	// Compute the convolution using inner_product and transform
	for (int i = 0; i < size_out; i++) {
		float sum = std::inner_product(input1_padded.begin() + i, input1_padded.begin() + i + size2, reversed_input2.begin(), 0.0);
		input1[i] = sum;
	}
}



void sawtooth_filter(std::array<std::array<float, 257>, 192>& data, std::array<std::array<float, 222>, 257>& scratch, std::array<std::array<float, 257>, 192>& output, int& NBINS) {
	// Initialize scratch to all zeros

	scratch.fill({ 0.0 });

	static const std::vector<float> filter = { 0.0, 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.0, 0.85714286, 0.71428571, 0.57142857, 0.42857143, 0.28571429, 0.14285714, 0.0 };

	// Transpose data into scratch
	for (int i = 0; i < 192; i++) {
		for (int j = 0; j < NBINS; j++) {
			scratch[j][i + 15] = data[i][j];
		}
	}

	// Create dynamic buffer to hold flattened scratch std::array
	//this is the only place we use a vector anywhere in the entire program, and that's because we cannot know exactly how many NBINS the user wants to process.
	std::vector<float> buffer(NBINS * 207, 0.0);

	// Flatten scratch std::array into buffer
	for (auto i = 0; i < NBINS; i++) {
		std::copy(scratch[i].begin() + 15, scratch[i].begin() + 222, buffer.begin() + (i * 207));
	}

	// Call same_mode_convolve
	same_vector_convolve(buffer, filter);

	// Transpose buffer into output
	for (int i = 0; i < NBINS; i++) {
		for (int j = 0; j < 192; j++) {
			output[j][i] = buffer[i * 207 + j + 15];
		}
	}
}



# define M_PI           3.14159265358979323846  /* pi */


class  Filter
{
private:
	//Filter Class Commentary
	//Stack size is ~200kb. We are not going to use the heap and play dangle the heap.
	//Everything is either allocated within an(external) function, passed by value as input to our function, 
	//everything instantated in Filter::Data is automatically destroyed when filter is destroyed, so we dont have to worry about smashing the stack.
	//this gives us good performance, minimum complexity, and minimum running around the smoke pile.
	//everything is either an int or a float by design. We never exceed an int value of 57054, so we will never need size_t.
	//we never manage anything but our static, fixed size std::arrays, so we will never need anything else! c++ is garbage, use python.
	// This code needs to be portable to every other language, and easy to understand, without needing to understand language specific shenanigans.

	//You just pass 8192 floats into the class, and get 8192 back. that's it.
	//it only works at a sample rate of 48000. Anything else and I can't guarantee anything.
	//you can pass a different entropy constant and NBINS. 

	static constexpr std::array<float, 512> shifted_logistic_window{ 1., 0.94409333, 0.88818667, 0.85535081, 0.83195987, 0.81374363, 0.79880003, 0.78611443, 0.77508119, 0.76530966, 0.75653308, 0.7485612, 0.74125354, 0.73450341, 0.72822787, 0.72236118, 0.71685033, 0.71165198, 0.7067302, 0.70205491, 0.69760064, 0.69334567, 0.68927131, 0.68536134, 0.68160163, 0.67797977, 0.67448482, 0.67110707, 0.66783786, 0.66466946, 0.6615949, 0.65860793, 0.65570285, 0.65287452, 0.65011824, 0.64742972, 0.64480505, 0.64224061, 0.63973308, 0.63727941, 0.63487677, 0.63252254, 0.63021427, 0.62794972, 0.62572677, 0.62354345, 0.62139793, 0.61928848, 0.6172135, 0.61517145, 0.61316093, 0.61118059, 0.60922916, 0.60730545, 0.60540833, 0.60353673, 0.60168964, 0.59986611, 0.59806522, 0.5962861, 0.59452794, 0.59278996, 0.5910714, 0.58937155, 0.58768973, 0.5860253, 0.58437763, 0.58274613, 0.58113023, 0.57952939, 0.57794307, 0.57637077, 0.57481202, 0.57326635, 0.5717333, 0.57021245, 0.56870339, 0.56720571, 0.56571903, 0.56424297, 0.56277718, 0.56132129, 0.55987498, 0.55843793, 0.5570098, 0.55559029, 0.55417912, 0.55277598, 0.55138059, 0.54999269, 0.548612, 0.54723826, 0.54587123, 0.54451066, 0.5431563, 0.54180793, 0.54046531, 0.53912822, 0.53779645, 0.53646977, 0.53514798, 0.53383087, 0.53251824, 0.53120989, 0.52990563, 0.52860527, 0.52730862, 0.5260155, 0.52472572, 0.5234391, 0.52215548, 0.52087466, 0.51959649, 0.51832079, 0.51704739, 0.51577613, 0.51450685, 0.51323937, 0.51197354, 0.5107092, 0.50944618, 0.50818433, 0.5069235, 0.50566352, 0.50440424, 0.5031455, 0.50188714, 0.50062902, 0.49937098, 0.49811286, 0.4968545, 0.49559576, 0.49433648, 0.4930765, 0.49181567, 0.49055382, 0.4892908, 0.48802646, 0.48676063, 0.48549315, 0.48422387, 0.48295261, 0.48167921, 0.48040351, 0.47912534, 0.47784452, 0.4765609, 0.47527428, 0.4739845, 0.47269138, 0.47139473, 0.47009437, 0.46879011, 0.46748176, 0.46616913, 0.46485202, 0.46353023, 0.46220355, 0.46087178, 0.45953469, 0.45819207, 0.4568437, 0.45548934, 0.45412877, 0.45276174, 0.451388, 0.45000731, 0.44861941, 0.44722402, 0.44582088, 0.44440971, 0.4429902, 0.44156207, 0.44012502, 0.43867871, 0.43722282, 0.43575703, 0.43428097, 0.43279429, 0.43129661, 0.42978755, 0.4282667, 0.42673365, 0.42518798, 0.42362923, 0.42205693, 0.42047061, 0.41886977, 0.41725387, 0.41562237, 0.4139747, 0.41231027, 0.41062845, 0.4089286, 0.40721004, 0.40547206, 0.4037139, 0.40193478, 0.40013389, 0.39831036, 0.39646327, 0.39459167, 0.39269455, 0.39077084, 0.38881941, 0.38683907, 0.38482855, 0.3827865, 0.38071152, 0.37860207, 0.37645655, 0.37427323, 0.37205028, 0.36978573, 0.36747746, 0.36512323, 0.36272059, 0.36026692, 0.35775939, 0.35519495, 0.35257028, 0.34988176, 0.34712548, 0.34429715, 0.34139207, 0.3384051, 0.33533054, 0.33216214, 0.32889293, 0.32551518, 0.32202023, 0.31839837, 0.31463866, 0.31072869, 0.30665433, 0.30239936, 0.29794509, 0.2932698, 0.28834802, 0.28314967, 0.27763882, 0.27177213, 0.26549659, 0.25874646, 0.2514388, 0.24346692, 0.23469034, 0.22491881, 0.21388557, 0.20119997, 0.18625637, 0.16804013, 0.14464919, 0.11181333, 0.05590667, 0., 0., 0.05590667, 0.11181333, 0.14464919, 0.16804013, 0.18625637, 0.20119997, 0.21388557, 0.22491881, 0.23469034, 0.24346692, 0.2514388, 0.25874646, 0.26549659, 0.27177213, 0.27763882, 0.28314967, 0.28834802, 0.2932698, 0.29794509, 0.30239936, 0.30665433, 0.31072869, 0.31463866, 0.31839837, 0.32202023, 0.32551518, 0.32889293, 0.33216214, 0.33533054, 0.3384051, 0.34139207, 0.34429715, 0.34712548, 0.34988176, 0.35257028, 0.35519495, 0.35775939, 0.36026692, 0.36272059, 0.36512323, 0.36747746, 0.36978573, 0.37205028, 0.37427323, 0.37645655, 0.37860207, 0.38071152, 0.3827865, 0.38482855, 0.38683907, 0.38881941, 0.39077084, 0.39269455, 0.39459167, 0.39646327, 0.39831036, 0.40013389, 0.40193478, 0.4037139, 0.40547206, 0.40721004, 0.4089286, 0.41062845, 0.41231027, 0.4139747, 0.41562237, 0.41725387, 0.41886977, 0.42047061, 0.42205693, 0.42362923, 0.42518798, 0.42673365, 0.4282667, 0.42978755, 0.43129661, 0.43279429, 0.43428097, 0.43575703, 0.43722282, 0.43867871, 0.44012502, 0.44156207, 0.4429902, 0.44440971, 0.44582088, 0.44722402, 0.44861941, 0.45000731, 0.451388, 0.45276174, 0.45412877, 0.45548934, 0.4568437, 0.45819207, 0.45953469, 0.46087178, 0.46220355, 0.46353023, 0.46485202, 0.46616913, 0.46748176, 0.46879011, 0.47009437, 0.47139473, 0.47269138, 0.4739845, 0.47527428, 0.4765609, 0.47784452, 0.47912534, 0.48040351, 0.48167921, 0.48295261, 0.48422387, 0.48549315, 0.48676063, 0.48802646, 0.4892908, 0.49055382, 0.49181567, 0.4930765, 0.49433648, 0.49559576, 0.4968545, 0.49811286, 0.49937098, 0.50062902, 0.50188714, 0.5031455, 0.50440424, 0.50566352, 0.5069235, 0.50818433, 0.50944618, 0.5107092, 0.51197354, 0.51323937, 0.51450685, 0.51577613, 0.51704739, 0.51832079, 0.51959649, 0.52087466, 0.52215548, 0.5234391, 0.52472572, 0.5260155, 0.52730862, 0.52860527, 0.52990563, 0.53120989, 0.53251824, 0.53383087, 0.53514798, 0.53646977, 0.53779645, 0.53912822, 0.54046531, 0.54180793, 0.5431563, 0.54451066, 0.54587123, 0.54723826, 0.548612, 0.54999269, 0.55138059, 0.55277598, 0.55417912, 0.55559029, 0.5570098, 0.55843793, 0.55987498, 0.56132129, 0.56277718, 0.56424297, 0.56571903, 0.56720571, 0.56870339, 0.57021245, 0.5717333, 0.57326635, 0.57481202, 0.57637077, 0.57794307, 0.57952939, 0.58113023, 0.58274613, 0.58437763, 0.5860253, 0.58768973, 0.58937155, 0.5910714, 0.59278996, 0.59452794, 0.5962861, 0.59806522, 0.59986611, 0.60168964, 0.60353673, 0.60540833, 0.60730545, 0.60922916, 0.61118059, 0.61316093, 0.61517145, 0.6172135, 0.61928848, 0.62139793, 0.62354345, 0.62572677, 0.62794972, 0.63021427, 0.63252254, 0.63487677, 0.63727941, 0.63973308, 0.64224061, 0.64480505, 0.64742972, 0.65011824, 0.65287452, 0.65570285, 0.65860793, 0.6615949, 0.66466946, 0.66783786, 0.67110707, 0.67448482, 0.67797977, 0.68160163, 0.68536134, 0.68927131, 0.69334567, 0.69760064, 0.70205491, 0.7067302, 0.71165198, 0.71685033, 0.72236118, 0.72822787, 0.73450341, 0.74125354, 0.7485612, 0.75653308, 0.76530966, 0.77508119, 0.78611443, 0.79880003, 0.81374363, 0.83195987, 0.85535081, 0.88818667, 0.94409333, 1. };
	static constexpr std::array<float, 512> shifted_hann_window = { 9.99990551e-01, 9.99914959e-01, 9.99763787e-01, 9.99537058e-01, 9.99234805e-01, 9.98857075e-01, 9.98403924e-01, 9.97875422e-01, 9.97271648e-01, 9.96592693e-01, 9.95838660e-01, 9.95009663e-01, 9.94105827e-01, 9.93127290e-01, 9.92074198e-01, 9.90946711e-01, 9.89745000e-01, 9.88469246e-01, 9.87119643e-01, 9.85696393e-01, 9.84199713e-01, 9.82629829e-01, 9.80986977e-01, 9.79271407e-01, 9.77483377e-01, 9.75623159e-01, 9.73691033e-01, 9.71687291e-01, 9.69612237e-01, 9.67466184e-01, 9.65249456e-01, 9.62962389e-01, 9.60605328e-01, 9.58178630e-01, 9.55682662e-01, 9.53117800e-01, 9.50484434e-01, 9.47782960e-01, 9.45013788e-01, 9.42177336e-01, 9.39274033e-01, 9.36304317e-01, 9.33268638e-01, 9.30167455e-01, 9.27001237e-01, 9.23770461e-01, 9.20475618e-01, 9.17117204e-01, 9.13695728e-01, 9.10211707e-01, 9.06665667e-01, 9.03058145e-01, 8.99389686e-01, 8.95660845e-01, 8.91872186e-01, 8.88024281e-01, 8.84117711e-01, 8.80153069e-01, 8.76130952e-01, 8.72051970e-01, 8.67916738e-01, 8.63725883e-01, 8.59480037e-01, 8.55179843e-01, 8.50825950e-01, 8.46419017e-01, 8.41959711e-01, 8.37448705e-01, 8.32886681e-01, 8.28274329e-01, 8.23612347e-01, 8.18901439e-01, 8.14142317e-01, 8.09335702e-01, 8.04482319e-01, 7.99582902e-01, 7.94638193e-01, 7.89648938e-01, 7.84615893e-01, 7.79539817e-01, 7.74421479e-01, 7.69261652e-01, 7.64061117e-01, 7.58820659e-01, 7.53541070e-01, 7.48223150e-01, 7.42867702e-01, 7.37475536e-01, 7.32047467e-01, 7.26584315e-01, 7.21086907e-01, 7.15556073e-01, 7.09992651e-01, 7.04397480e-01, 6.98771407e-01, 6.93115283e-01, 6.87429962e-01, 6.81716305e-01, 6.75975174e-01, 6.70207439e-01, 6.64413970e-01, 6.58595644e-01, 6.52753341e-01, 6.46887944e-01, 6.41000339e-01, 6.35091417e-01, 6.29162070e-01, 6.23213197e-01, 6.17245695e-01, 6.11260467e-01, 6.05258418e-01, 5.99240456e-01, 5.93207489e-01, 5.87160431e-01, 5.81100196e-01, 5.75027699e-01, 5.68943859e-01, 5.62849596e-01, 5.56745831e-01, 5.50633486e-01, 5.44513487e-01, 5.38386758e-01, 5.32254225e-01, 5.26116815e-01, 5.19975458e-01, 5.13831080e-01, 5.07684611e-01, 5.01536980e-01, 4.95389117e-01, 4.89241951e-01, 4.83096412e-01, 4.76953428e-01, 4.70813928e-01, 4.64678841e-01, 4.58549094e-01, 4.52425614e-01, 4.46309327e-01, 4.40201156e-01, 4.34102027e-01, 4.28012860e-01, 4.21934577e-01, 4.15868096e-01, 4.09814335e-01, 4.03774209e-01, 3.97748631e-01, 3.91738511e-01, 3.85744760e-01, 3.79768282e-01, 3.73809982e-01, 3.67870760e-01, 3.61951513e-01, 3.56053138e-01, 3.50176526e-01, 3.44322565e-01, 3.38492141e-01, 3.32686134e-01, 3.26905422e-01, 3.21150881e-01, 3.15423378e-01, 3.09723782e-01, 3.04052952e-01, 2.98411747e-01, 2.92801019e-01, 2.87221617e-01, 2.81674384e-01, 2.76160159e-01, 2.70679775e-01, 2.65234062e-01, 2.59823842e-01, 2.54449933e-01, 2.49113148e-01, 2.43814294e-01, 2.38554171e-01, 2.33333576e-01, 2.28153297e-01, 2.23014117e-01, 2.17916814e-01, 2.12862158e-01, 2.07850913e-01, 2.02883837e-01, 1.97961681e-01, 1.93085190e-01, 1.88255099e-01, 1.83472140e-01, 1.78737036e-01, 1.74050502e-01, 1.69413247e-01, 1.64825973e-01, 1.60289372e-01, 1.55804131e-01, 1.51370928e-01, 1.46990432e-01, 1.42663307e-01, 1.38390206e-01, 1.34171776e-01, 1.30008654e-01, 1.25901469e-01, 1.21850843e-01, 1.17857388e-01, 1.13921708e-01, 1.10044397e-01, 1.06226043e-01, 1.02467221e-01, 9.87685015e-02, 9.51304424e-02, 9.15535940e-02, 8.80384971e-02, 8.45856832e-02, 8.11956742e-02, 7.78689827e-02, 7.46061116e-02, 7.14075543e-02, 6.82737943e-02, 6.52053053e-02, 6.22025514e-02, 5.92659864e-02, 5.63960544e-02, 5.35931893e-02, 5.08578147e-02, 4.81903443e-02, 4.55911813e-02, 4.30607187e-02, 4.05993391e-02, 3.82074146e-02, 3.58853068e-02, 3.36333667e-02, 3.14519350e-02, 2.93413412e-02, 2.73019047e-02, 2.53339336e-02, 2.34377255e-02, 2.16135671e-02, 1.98617342e-02, 1.81824916e-02, 1.65760932e-02, 1.50427819e-02, 1.35827895e-02, 1.21963367e-02, 1.08836332e-02, 9.64487731e-03, 8.48025644e-03, 7.38994662e-03, 6.37411270e-03, 5.43290826e-03, 4.56647559e-03, 3.77494569e-03, 3.05843822e-03, 2.41706151e-03, 1.85091253e-03, 1.36007687e-03, 9.44628746e-04, 6.04630957e-04, 3.40134910e-04, 1.51180595e-04, 3.77965773e-05, 0.00000000e+00, 0.00000000e+00, 3.77965773e-05, 1.51180595e-04, 3.40134910e-04, 6.04630957e-04, 9.44628746e-04, 1.36007687e-03, 1.85091253e-03, 2.41706151e-03, 3.05843822e-03, 3.77494569e-03, 4.56647559e-03, 5.43290826e-03, 6.37411270e-03, 7.38994662e-03, 8.48025644e-03, 9.64487731e-03, 1.08836332e-02, 1.21963367e-02, 1.35827895e-02, 1.50427819e-02, 1.65760932e-02, 1.81824916e-02, 1.98617342e-02, 2.16135671e-02, 2.34377255e-02, 2.53339336e-02, 2.73019047e-02, 2.93413412e-02, 3.14519350e-02, 3.36333667e-02, 3.58853068e-02, 3.82074146e-02, 4.05993391e-02, 4.30607187e-02, 4.55911813e-02, 4.81903443e-02, 5.08578147e-02, 5.35931893e-02, 5.63960544e-02, 5.92659864e-02, 6.22025514e-02, 6.52053053e-02, 6.82737943e-02, 7.14075543e-02, 7.46061116e-02, 7.78689827e-02, 8.11956742e-02, 8.45856832e-02, 8.80384971e-02, 9.15535940e-02, 9.51304424e-02, 9.87685015e-02, 1.02467221e-01, 1.06226043e-01, 1.10044397e-01, 1.13921708e-01, 1.17857388e-01, 1.21850843e-01, 1.25901469e-01, 1.30008654e-01, 1.34171776e-01, 1.38390206e-01, 1.42663307e-01, 1.46990432e-01, 1.51370928e-01, 1.55804131e-01, 1.60289372e-01, 1.64825973e-01, 1.69413247e-01, 1.74050502e-01, 1.78737036e-01, 1.83472140e-01, 1.88255099e-01, 1.93085190e-01, 1.97961681e-01, 2.02883837e-01, 2.07850913e-01, 2.12862158e-01, 2.17916814e-01, 2.23014117e-01, 2.28153297e-01, 2.33333576e-01, 2.38554171e-01, 2.43814294e-01, 2.49113148e-01, 2.54449933e-01, 2.59823842e-01, 2.65234062e-01, 2.70679775e-01, 2.76160159e-01, 2.81674384e-01, 2.87221617e-01, 2.92801019e-01, 2.98411747e-01, 3.04052952e-01, 3.09723782e-01, 3.15423378e-01, 3.21150881e-01, 3.26905422e-01, 3.32686134e-01, 3.38492141e-01, 3.44322565e-01, 3.50176526e-01, 3.56053138e-01, 3.61951513e-01, 3.67870760e-01, 3.73809982e-01, 3.79768282e-01, 3.85744760e-01, 3.91738511e-01, 3.97748631e-01, 4.03774209e-01, 4.09814335e-01, 4.15868096e-01, 4.21934577e-01, 4.28012860e-01, 4.34102027e-01, 4.40201156e-01, 4.46309327e-01, 4.52425614e-01, 4.58549094e-01, 4.64678841e-01, 4.70813928e-01, 4.76953428e-01, 4.83096412e-01, 4.89241951e-01, 4.95389117e-01, 5.01536980e-01, 5.07684611e-01, 5.13831080e-01, 5.19975458e-01, 5.26116815e-01, 5.32254225e-01, 5.38386758e-01, 5.44513487e-01, 5.50633486e-01, 5.56745831e-01, 5.62849596e-01, 5.68943859e-01, 5.75027699e-01, 5.81100196e-01, 5.87160431e-01, 5.93207489e-01, 5.99240456e-01, 6.05258418e-01, 6.11260467e-01, 6.17245695e-01, 6.23213197e-01, 6.29162070e-01, 6.35091417e-01, 6.41000339e-01, 6.46887944e-01, 6.52753341e-01, 6.58595644e-01, 6.64413970e-01, 6.70207439e-01, 6.75975174e-01, 6.81716305e-01, 6.87429962e-01, 6.93115283e-01, 6.98771407e-01, 7.04397480e-01, 7.09992651e-01, 7.15556073e-01, 7.21086907e-01, 7.26584315e-01, 7.32047467e-01, 7.37475536e-01, 7.42867702e-01, 7.48223150e-01, 7.53541070e-01, 7.58820659e-01, 7.64061117e-01, 7.69261652e-01, 7.74421479e-01, 7.79539817e-01, 7.84615893e-01, 7.89648938e-01, 7.94638193e-01, 7.99582902e-01, 8.04482319e-01, 8.09335702e-01, 8.14142317e-01, 8.18901439e-01, 8.23612347e-01, 8.28274329e-01, 8.32886681e-01, 8.37448705e-01, 8.41959711e-01, 8.46419017e-01, 8.50825950e-01, 8.55179843e-01, 8.59480037e-01, 8.63725883e-01, 8.67916738e-01, 8.72051970e-01, 8.76130952e-01, 8.80153069e-01, 8.84117711e-01, 8.88024281e-01, 8.91872186e-01, 8.95660845e-01, 8.99389686e-01, 9.03058145e-01, 9.06665667e-01, 9.10211707e-01, 9.13695728e-01, 9.17117204e-01, 9.20475618e-01, 9.23770461e-01, 9.27001237e-01, 9.30167455e-01, 9.33268638e-01, 9.36304317e-01, 9.39274033e-01, 9.42177336e-01, 9.45013788e-01, 9.47782960e-01, 9.50484434e-01, 9.53117800e-01, 9.55682662e-01, 9.58178630e-01, 9.60605328e-01, 9.62962389e-01, 9.65249456e-01, 9.67466184e-01, 9.69612237e-01, 9.71687291e-01, 9.73691033e-01, 9.75623159e-01, 9.77483377e-01, 9.79271407e-01, 9.80986977e-01, 9.82629829e-01, 9.84199713e-01, 9.85696393e-01, 9.87119643e-01, 9.88469246e-01, 9.89745000e-01, 9.90946711e-01, 9.92074198e-01, 9.93127290e-01, 9.94105827e-01, 9.95009663e-01, 9.95838660e-01, 9.96592693e-01, 9.97271648e-01, 9.97875422e-01, 9.98403924e-01, 9.98857075e-01, 9.99234805e-01, 9.99537058e-01, 9.99763787e-01, 9.99914959e-01, 9.99990551e-01 };
	static constexpr std::array<float, 512> synthesis_window = { 0.00000000e+00,2.52493737e-05,1.00993617e-04,2.27221124e-04,4.03912573e-04,6.31040943e-04,9.08571512e-04,1.23646188e-03,1.61466197e-03,2.04311406e-03,2.52175277e-03,3.05050510e-03,3.62929044e-03,4.25802059e-03,4.93659976e-03,5.66492464e-03,6.44288435e-03,7.27036053e-03,8.14722732e-03,9.07335139e-03,1.00485920e-02,1.10728009e-02,1.21458227e-02,1.32674943e-02,1.44376456e-02,1.56560990e-02,1.69226697e-02,1.82371657e-02,1.95993878e-02,2.10091294e-02,2.24661772e-02,2.39703105e-02,2.55213015e-02,2.71189155e-02,2.87629109e-02,3.04530390e-02,3.21890442e-02,3.39706641e-02,3.57976294e-02,3.76696640e-02,3.95864853e-02,4.15478036e-02,4.35533228e-02,4.56027403e-02,4.76957466e-02,4.98320260e-02,5.20112561e-02,5.42331081e-02,5.64972471e-02,5.88033315e-02,6.11510136e-02,6.35399396e-02,6.59697493e-02,6.84400765e-02,7.09505488e-02,7.35007880e-02,7.60904099e-02,7.87190242e-02,8.13862349e-02,8.40916401e-02,8.68348324e-02,8.96153985e-02,9.24329196e-02,9.52869711e-02,9.81771232e-02,1.01102941e-01,1.04063982e-01,1.07059803e-01,1.10089950e-01,1.13153968e-01,1.16251394e-01,1.19381763e-01,1.22544602e-01,1.25739435e-01,1.28965780e-01,1.32223151e-01,1.35511057e-01,1.38829001e-01,1.42176485e-01,1.45553002e-01,1.48958043e-01,1.52391095e-01,1.55851640e-01,1.59339154e-01,1.62853113e-01,1.66392985e-01,1.69958235e-01,1.73548325e-01,1.77162713e-01,1.80800852e-01,1.84462192e-01,1.88146180e-01,1.91852259e-01,1.95579867e-01,1.99328441e-01,2.03097413e-01,2.06886212e-01,2.10694266e-01,2.14520996e-01,2.18365823e-01,2.22228164e-01,2.26107433e-01,2.30003042e-01,2.33914400e-01,2.37840913e-01,2.41781985e-01,2.45737016e-01,2.49705407e-01,2.53686555e-01,2.57679853e-01,2.61684694e-01,2.65700470e-01,2.69726568e-01,2.73762377e-01,2.77807281e-01,2.81860664e-01,2.85921908e-01,2.89990394e-01,2.94065503e-01,2.98146611e-01,3.02233096e-01,3.06324335e-01,3.10419702e-01,3.14518572e-01,3.18620319e-01,3.22724316e-01,3.26829934e-01,3.30936547e-01,3.35043526e-01,3.39150246e-01,3.43256086e-01,3.47360427e-01,3.51462648e-01,3.55562129e-01,3.59658251e-01,3.63750398e-01,3.67837950e-01,3.71920292e-01,3.75996808e-01,3.80066883e-01,3.84129905e-01,3.88185260e-01,3.92232339e-01,3.96270531e-01,4.00299230e-01,4.04317828e-01,4.08325721e-01,4.12322306e-01,4.16306982e-01,4.20279150e-01,4.24238213e-01,4.28183575e-01,4.32114645e-01,4.36030831e-01,4.39931546e-01,4.43816203e-01,4.47684220e-01,4.51535015e-01,4.55368011e-01,4.59182632e-01,4.62978306e-01,4.66754464e-01,4.70510539e-01,4.74245967e-01,4.77960189e-01,4.81652647e-01,4.85322788e-01,4.88970060e-01,4.92593918e-01,4.96193817e-01,4.99769218e-01,5.03319584e-01,5.06844384e-01,5.10343088e-01,5.13815172e-01,5.17260116e-01,5.20677401e-01,5.24066516e-01,5.27426952e-01,5.30758205e-01,5.34059774e-01,5.37331165e-01,5.40571886e-01,5.43781450e-01,5.46959376e-01,5.50105185e-01,5.53218405e-01,5.56298569e-01,5.59345212e-01,5.62357878e-01,5.65336111e-01,5.68279465e-01,5.71187495e-01,5.74059764e-01,5.76895840e-01,5.79695293e-01,5.82457703e-01,5.85182652e-01,5.87869728e-01,5.90518527e-01,5.93128647e-01,5.95699693e-01,5.98231278e-01,6.00723016e-01,6.03174532e-01,6.05585452e-01,6.07955411e-01,6.10284050e-01,6.12571014e-01,6.14815956e-01,6.17018534e-01,6.19178413e-01,6.21295262e-01,6.23368760e-01,6.25398589e-01,6.27384439e-01,6.29326006e-01,6.31222993e-01,6.33075109e-01,6.34882068e-01,6.36643595e-01,6.38359416e-01,6.40029269e-01,6.41652895e-01,6.43230043e-01,6.44760469e-01,6.46243937e-01,6.47680215e-01,6.49069080e-01,6.50410317e-01,6.51703715e-01,6.52949072e-01,6.54146194e-01,6.55294893e-01,6.56394987e-01,6.57446303e-01,6.58448674e-01,6.59401943e-01,6.60305956e-01,6.61160570e-01,6.61965647e-01,6.62721059e-01,6.63426683e-01,6.64082404e-01,6.64688115e-01,6.65243717e-01,6.65749117e-01,6.66204231e-01,6.66608983e-01,6.66963302e-01,6.67267128e-01,6.67520406e-01,6.67723090e-01,6.67875141e-01,6.67976529e-01,6.68027230e-01,6.68027230e-01,6.67976529e-01,6.67875141e-01,6.67723090e-01,6.67520406e-01,6.67267128e-01,6.66963302e-01,6.66608983e-01,6.66204231e-01,6.65749117e-01,6.65243717e-01,6.64688115e-01,6.64082404e-01,6.63426683e-01,6.62721059e-01,6.61965647e-01,6.61160570e-01,6.60305956e-01,6.59401943e-01,6.58448674e-01,6.57446303e-01,6.56394987e-01,6.55294893e-01,6.54146194e-01,6.52949072e-01,6.51703715e-01,6.50410317e-01,6.49069080e-01,6.47680215e-01,6.46243937e-01,6.44760469e-01,6.43230043e-01,6.41652895e-01,6.40029269e-01,6.38359416e-01,6.36643595e-01,6.34882068e-01,6.33075109e-01,6.31222993e-01,6.29326006e-01,6.27384439e-01,6.25398589e-01,6.23368760e-01,6.21295262e-01,6.19178413e-01,6.17018534e-01,6.14815956e-01,6.12571014e-01,6.10284050e-01,6.07955411e-01,6.05585452e-01,6.03174532e-01,6.00723016e-01,5.98231278e-01,5.95699693e-01,5.93128647e-01,5.90518527e-01,5.87869728e-01,5.85182652e-01,5.82457703e-01,5.79695293e-01,5.76895840e-01,5.74059764e-01,5.71187495e-01,5.68279465e-01,5.65336111e-01,5.62357878e-01,5.59345212e-01,5.56298569e-01,5.53218405e-01,5.50105185e-01,5.46959376e-01,5.43781450e-01,5.40571886e-01,5.37331165e-01,5.34059774e-01,5.30758205e-01,5.27426952e-01,5.24066516e-01,5.20677401e-01,5.17260116e-01,5.13815172e-01,5.10343088e-01,5.06844384e-01,5.03319584e-01,4.99769218e-01,4.96193817e-01,4.92593918e-01,4.88970060e-01,4.85322788e-01,4.81652647e-01,4.77960189e-01,4.74245967e-01,4.70510539e-01,4.66754464e-01,4.62978306e-01,4.59182632e-01,4.55368011e-01,4.51535015e-01,4.47684220e-01,4.43816203e-01,4.39931546e-01,4.36030831e-01,4.32114645e-01,4.28183575e-01,4.24238213e-01,4.20279150e-01,4.16306982e-01,4.12322306e-01,4.08325721e-01,4.04317828e-01,4.00299230e-01,3.96270531e-01,3.92232339e-01,3.88185260e-01,3.84129905e-01,3.80066883e-01,3.75996808e-01,3.71920292e-01,3.67837950e-01,3.63750398e-01,3.59658251e-01,3.55562129e-01,3.51462648e-01,3.47360427e-01,3.43256086e-01,3.39150246e-01,3.35043526e-01,3.30936547e-01,3.26829934e-01,3.22724316e-01,3.18620319e-01,3.14518572e-01,3.10419702e-01,3.06324335e-01,3.02233096e-01,2.98146611e-01,2.94065503e-01,2.89990394e-01,2.85921908e-01,2.81860664e-01,2.77807281e-01,2.73762377e-01,2.69726568e-01,2.65700470e-01,2.61684694e-01,2.57679853e-01,2.53686555e-01,2.49705407e-01,2.45737016e-01,2.41781985e-01,2.37840913e-01,2.33914400e-01,2.30003042e-01,2.26107433e-01,2.22228164e-01,2.18365823e-01,2.14520996e-01,2.10694266e-01,2.06886212e-01,2.03097413e-01,1.99328441e-01,1.95579867e-01,1.91852259e-01,1.88146180e-01,1.84462192e-01,1.80800852e-01,1.77162713e-01,1.73548325e-01,1.69958235e-01,1.66392985e-01,1.62853113e-01,1.59339154e-01,1.55851640e-01,1.52391095e-01,1.48958043e-01,1.45553002e-01,1.42176485e-01,1.38829001e-01,1.35511057e-01,1.32223151e-01,1.28965780e-01,1.25739435e-01,1.22544602e-01,1.19381763e-01,1.16251394e-01,1.13153968e-01,1.10089950e-01,1.07059803e-01,1.04063982e-01,1.01102941e-01,9.81771232e-02,9.52869711e-02,9.24329196e-02,8.96153985e-02,8.68348324e-02,8.40916401e-02,8.13862349e-02,7.87190242e-02,7.60904099e-02,7.35007880e-02,7.09505488e-02,6.84400765e-02,6.59697493e-02,6.35399396e-02,6.11510136e-02,5.88033315e-02,5.64972471e-02,5.42331081e-02,5.20112561e-02,4.98320260e-02,4.76957466e-02,4.56027403e-02,4.35533228e-02,4.15478036e-02,3.95864853e-02,3.76696640e-02,3.57976294e-02,3.39706641e-02,3.21890442e-02,3.04530390e-02,2.87629109e-02,2.71189155e-02,2.55213015e-02,2.39703105e-02,2.24661772e-02,2.10091294e-02,1.95993878e-02,1.82371657e-02,1.69226697e-02,1.56560990e-02,1.44376456e-02,1.32674943e-02,1.21458227e-02,1.10728009e-02,1.00485920e-02,9.07335139e-03,8.14722732e-03,7.27036053e-03,6.44288435e-03,5.66492464e-03,4.93659976e-03,4.25802059e-03,3.62929044e-03,3.05050510e-03,2.52175277e-03,2.04311406e-03,1.61466197e-03,1.23646188e-03,9.08571512e-04,6.31040943e-04,4.03912573e-04,2.27221124e-04,1.00993617e-04,2.52493737e-05,0.00000000e+00 };
	static constexpr std::array<float, 37> logit_37 = { 0.0, 0.08441118, 0.16882236, 0.2197072, 0.25693163, 0.28672628, 0.31187091, 0.33385255, 0.35356306, 0.37158192, 0.38830914, 0.40403462, 0.41897721, 0.43330834, 0.44716693, 0.46066936, 0.47391649, 0.4869987, 0.5, 0.5130013, 0.52608351, 0.53933064, 0.55283307, 0.56669166, 0.58102279, 0.59596538, 0.61169086, 0.62841808, 0.64643694, 0.66614745, 0.68812909, 0.71327372, 0.74306837, 0.7802928, 0.83117764, 0.91558882, 1.0 };
	static constexpr std::array<float, 3> entropy_filter = { 1.0,1.0,1.0 };
	//working memory
	std::array<float, 24576> audio = {};
	std::array<float, 25087> audio_padded = {};
	std::array<float, 384> buffer = {};
	std::array<float, 192> entropy_unmasked = {};
	std::array<int, 192> entropy_thresholded = {};
	std::array<float, 204> entropy_padded = {}; //192 + 12
	std::array<float, 257> logit_distribution = {};


	std::array<float, 8192> output = { 0 };
	std::array<float, 8192> empty = { 0 };//just a bunch of zeros


	float t=0, initial=0, multiplier=0, MAXIMUM= 0.6122169028112414, constant=0, CONST = 0.057,test=0, thresh1 = 0.0;
	int flag=0, count=0, NBINS_last=0, truecount = 0;
	bool set = true;


	std::array<float, 257> temp_257 = {};
	std::array<float, 512> temp_512 = {};
	std::array<std::array<float, 222>, 257> scratch = {};
	std::array<std::array<std::complex<float>, 257>, 192> stft_complex = {};
	std::array<std::array<std::complex<float>, 257>, 64> stft_output = {};
	std::array<std::array<std::complex<float>, 257>, 64> stft_zeros = {}; // for use in the scenario two where we're processing the residual buffer but without the audio
	std::array<std::array<float, 257>, 192> stft_real = {};
	std::array<std::array<float, 257>, 192> smoothed = {};
	std::array<std::array<float, 257>, 192> previous = {};


	// Define the lookup tables
	std::array<float, 66049> sin_table = {}; //257 x 257
	std::array<float, 66049> cos_table = {};
	//fill the tables

	inline constexpr void init_tables() {
		float angle = 0.0;
		for (int k = 0; k < 257; k++) {
			for (int n = 0; n < 257; n++) {
				angle = 2 * M_PI * k * n / 257.0;
				sin_table[((1 + k) * (1 + n)) - 1] = angle;
				cos_table[((1 + k) * (1 + n)) - 1] = angle;
			}
		}
		for (int k = 0; k < 66049; k++) {
			sin_table[k] = std::sin(sin_table[k]);
			cos_table[k] = std::sin(cos_table[k]);

		}
	}

	inline void find_max(const std::array<std::array<float, 257>, 192>& data, float& maximum) {
		maximum = 0;
		for (const auto& row : data) {
			for (int j = 0; j < NBINS_last; j++) {
				if (row[j] > maximum) {
					maximum = row[j];
				}
			}
		}
	}
	

	inline float correlationCoefficient(const std::array<float, 257>& X, const std::array<float, 257>& Y) {
		float sum_X = accumulate(X.begin(), X.begin() + NBINS_last, 0.0);
		float sum_Y = accumulate(Y.begin(), Y.begin() + NBINS_last, 0.0);
		float sum_XY = inner_product(X.begin(), X.begin() + NBINS_last, Y.begin(), 0.0);
		float squareSum_X = inner_product(X.begin(), X.begin() + NBINS_last, X.begin(), 0.0);
		float squareSum_Y = inner_product(Y.begin(), Y.begin() + NBINS_last, Y.begin(), 0.0);

		float corr = (NBINS_last * sum_XY - sum_X * sum_Y) /
			sqrt((NBINS_last * squareSum_X - sum_X * sum_X) *
				(NBINS_last * squareSum_Y - sum_Y * sum_Y));

		return corr;
	}

	inline void generate_true_logistic() {
		logit_distribution.fill(0.0);
		if (NBINS_last < 4) {
			return; //note: we don't generate logistics smaller than four points

			for (int i = 0; i < NBINS_last; i++) {
				logit_distribution[i] = static_cast<float>(i) / (NBINS_last - 1);
			}
			for (int i = 1; i < NBINS_last - 1; i++) {
				logit_distribution[i] /= 1 - logit_distribution[i];
				logit_distribution[i] = std::log(logit_distribution[i]);
			}
			logit_distribution[NBINS_last - 1] = (2 * logit_distribution[NBINS_last - 2]) - logit_distribution[NBINS_last - 3];
			logit_distribution[0] = -logit_distribution[NBINS_last - 1];

			// Interpolate the elements in logit to between 0 and 1
			float min_val = logit_distribution[0];
			float max_val = logit_distribution[NBINS_last - 1];
			for (int i = 0; i < NBINS_last; i++) {
				logit_distribution[i] = (logit_distribution[i] - min_val) / (max_val - min_val);
			}
		}

	}
	inline void determine_entropy_maximum() {
		temp_257.fill({ 0 });
		temp_257[NBINS_last - 1] = 1.0;
		MAXIMUM = 1.0 - correlationCoefficient(temp_257, logit_distribution);
	}


	inline void fast_entropy(std::array<std::array<float, 257>, 192>& data) {

		for (int i = 0; i < 192; i++) {

			// Create a subset of the first NBINS elements of d and sort it
			copy(data[i].begin(), data[i].begin() + NBINS_last, temp_257.begin());
			sort(temp_257.begin(), temp_257.begin() + NBINS_last);

			float dx = temp_257[NBINS_last - 1] - temp_257[0];
			for (int j = 0; j < NBINS_last; j++) {
				temp_257[j] = (temp_257[j] - temp_257[0]) / dx;
			}

			float v = correlationCoefficient(temp_257, logit_distribution);
			if (isnan(v)) {
				entropy_unmasked[i] = 0;
			}
			else {
				entropy_unmasked[i] = 1.0 - v;
			}
		}
	}

	inline void fast_peaks(std::array<std::array<float, 257>, 192>& stft_, std::array<std::array<float, 257>, 192>& mask) {
		for (int each = 0; each < 192; each++) {
			if (entropy_thresholded[each] == 0) {
				continue; //skip the calculations for this row, it's masked already
			}
			constant = MAN(stft_[each], NBINS_last);
			test = entropy_padded[each + 8] / MAXIMUM;
			test = abs(test - 1);
			thresh1 = (t * test);
			if (!isnan(thresh1)) {
				constant = (thresh1 + constant) / 2; //catch errors
			}
			for (int i = 0; i < NBINS_last; i++) {
				if (stft_[i][each] > CONST) {
					mask[i][each] = 1;
				}
			}
		}
	}

	inline void rfft(std::array<float, 257>& x, std::array<std::complex<float>, 257>& X) {

		// Compute the FFT of the positive frequencies
		for (int k = 0; k < 128; k++) {
			float real = 0;
			float imag = 0;

			// Loop over each sample index n in the input std::array x
			for (int n = 0; n < 257; n++) {
				// Compute the real and imaginary components of X[k]
				//angle consists of 32896 possible floats generated with 2* M_Pi * k(0...256) * n(0...256) / 257; 0.0...794.8596136460028
				//and storing it in index [k*n], and then obtaining it using retrieve_sin or retrieve_cos [k*n]

				real += x[n] * cos_table[((1 + k) * (1 + n)) - 1];
				imag -= x[n] * sin_table[((1 + k) * (1 + n)) - 1];
			}

			// Store the computed real and imaginary components in X[k]
			X[k] = std::complex<float>(real, imag);
		}

		// Compute the FFT of the negative frequencies by conjugating X[N/2 - k]
		for (int k = 1; k < 128; k++) {
			X[128 - k] = (X[k]);
		}
		X[128] = -std::conj(X[128]);
		X[256] = std::complex<float>(((2.0 * X[255].real()) - X[254].real()), ((2.0 * X[255].imag()) - X[254].imag())); //is this correct numpy behavior?
	}



	inline void stft(std::array<float, 25087>& xp, const std::array<float, 512>& window, std::array<std::array<std::complex<float>, 257>, 192>& out) {

		// Initialize Sx_temp to all zeros
		temp_512.fill({ 0 });

		for (int i = 0; i < 192; i++) {
			int start0 = 128 * i;
			int end0 = start0 + 256;
			int start1 = end0;

			for (int j = 0; j < 256; j++) {
				temp_512[j] = xp[start1 + j];
			}
			for (int j = 256; j < 512; j++) {
				temp_512[j] = xp[start0 + j - 256];
			}

			// Apply windowing and copy in one step
			for (int j = 0; j < 257; j++) {
				temp_257[j] = temp_512[j] * window[j];
			}

			// Perform in-place rfft
			rfft(temp_257, out[i]);
		}
	}


	inline void istft(std::array<std::array<std::complex<float>, 257>, 64>& Sx) {


		// Reuse temp, eliminate xbuf
		for (int i = 0; i < 64; i++) {
			// Perform in-place irfft and fftshift using temp and xbuf std::arrays
			for (int n = 0; n < 257; n++) {
				float real_part = 0;
				for (int k = 0; k < 257; k++) {
					//float angle = 2 * M_PI * k * n / 257;
					//a different lookup table is needed here
					//angle consists of 65536 possible floats generated with 2* M_Pi * k(0...256) * n(0...256) / 257, 0.0.. 1602.2367015226512
					//and storing it in index [k*n], and then obtaining it using retrieve_sin_inv(int k*n) or retrieve_cos_inv(int,k*n)

					real_part += Sx[i][k].real() * cos_table[((1 + k) * (1 + n)) - 1] - Sx[i][k].imag() * sin_table[((1 + k) * (1 + n)) - 1];
				}
				temp_257[n] = real_part / 257.0;
			}
			// Perform FFT shift for a fixed size 

			for (int i = 0; i < 128; i++) {
				std::swap(temp_257[i], temp_257[i + 128]);
			}

			int n = 0;
			for (int j = 0; j < 257; j++) {
				// Apply window and add to buffer
				buffer[j % 128] += temp_257[i] * synthesis_window[j];

				// Generate output samples and shift buffer left
				if (j % 128 == 0) {
					output[n] = buffer[0];
					buffer[0] = buffer[128];
					buffer[128] = 0.0;
					n += 128;
				}
			}
		}
	}

	inline int longestConsecutive(std::array<int, 192>& nums) {
		int curr_streak = 0;
		int prevstreak = 0;
		for (int i = 32; i < 160; i++) {
			if (nums[i] == 1) {
				curr_streak++;
			}
			else if (nums[i] == 0) {
				if (curr_streak > prevstreak) {
					prevstreak = curr_streak;
				}
				curr_streak = 0;
			}
		}
		return (curr_streak > prevstreak) ? curr_streak : prevstreak;
	}

	inline void remove_outliers(std::array<int, 192>& a, int value, int threshold, int replace) {
		int first = 0;
		int end = 0;
		int index = 0;
		while (first < 192) {
			if (a[first] == value) {
				index = first;
				while (index < 192 && a[index] == value) {
					index += 1;
				}
				end = index;
				if (end - first + 1 < threshold) {
					for (int i = first; i < end; i++) {
						a[i] = replace;
					}
				}
				first = end;
			}
			else {
				index = first;
				while (index < 192 && a[index] != value) {
					index += 1;
				}
				first = index;
			}
		}
	}


public:
	constexpr Filter() : sin_table(), cos_table(){
		init_tables();
		std::copy(std::begin(logit_37), std::end(logit_37), std::begin(logit_distribution));
	}


		//TODO: find ways to merge the uses of the above so that a mimimum in working memory can be utilized

	std::array<float, 8192> process(std::array<float, 8192> input, float CONST_1 = 0.057, int NBINS = 37) {
		if ((NBINS != NBINS_last)) { set = false; }//keep track of changes to what the function is called with


		if (set == false) {
			if (NBINS == 37) {//let's fill the std::array
				NBINS_last = NBINS;
				CONST = CONST_1;
				std::copy(std::begin(logit_37), std::end(logit_37), std::begin(logit_distribution));
				MAXIMUM = 0.6122169028112414;
				set = true;
			}
			else {
				//generate and set logit, entropy maximum
				NBINS_last = NBINS;
				CONST = CONST_1;
				generate_true_logistic();
				determine_entropy_maximum();
				set = true;

			}
		}


		//in this function we are only to allocate and do reference based manipulation of data- never copy or move. 
		rotate(begin(audio), begin(audio) + 8192, end(audio)); // Shift the values in the std::array 8192 values to the left.
		//rapidly copy in the audio
		std::copy(std::begin(input), std::end(input), std::end(audio) - 8192);

		copy(begin(input), end(input), end(audio) - 8192); // Copy the contents of input into the last 8192 elements of audio.
		for (int i = 256; i < 25087 - 255; i++) {
			audio_padded[i] = audio[i - 256];
		}
		for (int i = 0; i < 256; i++) {
			audio_padded[i] = audio[257 + 2 * 256 - i - 1];
		}
		for (int i = 0; i < 255; i++) {
			audio_padded[25087 - 255 + i] = audio[audio.size() - 256 - i - 1];
		}
		stft(audio_padded, shifted_logistic_window, stft_complex);
		// Copy the first 37 rows of stft_complex to stft_real
		for (int i = 0; i < 257; i++) {
			for (int j = 0; j < 192; j++) {
				stft_real[j][i] = abs(stft_complex[j][i]);
			}
		}
		fast_entropy(stft_real);

		for (int i = 0; i < 192; i++) {
			if (isnan(entropy_unmasked[i])) {
				entropy_unmasked[i] = 0;
			}
		}
		copy(begin(entropy_unmasked), end(entropy_unmasked), entropy_padded.begin() + 3); //copy into middle of padding
		for (int i = 0; i < 204; i++) {
			float sum = std::inner_product(entropy_padded.begin() + i, entropy_padded.begin() + i + 3, entropy_filter.begin(), 0.0);
			entropy_padded[i] = sum/3;
		} //perform inline convolution


		for (int i = 6; i < 198; i++) {
			if (entropy_padded[i] < constant) {
				entropy_thresholded[i - 8] = 0;
			}
			else {
				entropy_thresholded[i - 8] = 1;
				count++;
				if ((i > 38) && (i < 166)) {
					truecount++;
				}
			}
		}

		if (count > 0) {
			//initial criteria determined and entropy thresholded in one step.
			count = 0; //clear for next run

			if (truecount > 22 || longestConsecutive(entropy_thresholded) > 16) {
				truecount = 0;//clear for next run
				remove_outliers(entropy_thresholded, 0, 6, 1);
				remove_outliers(entropy_thresholded, 1, 2, 0);
				threshold(stft_real, NBINS, t);
				find_max(stft_real, initial);
				sawtooth_filter(stft_real, scratch, smoothed, NBINS);
				fast_peaks(smoothed, previous);

				for (int i = 0; i < NBINS; i++) {
					for (int j = 0; j < 192; j++) {
						if (previous[i][j] == 0) {
							stft_real[i][j] = 0.0;
						}
					}
				}

				find_max(stft_real, multiplier);
				multiplier = multiplier / initial;
				if (multiplier > 1) { multiplier = 1; }

				sawtooth_filter(stft_real, scratch, smoothed, NBINS);

				fast_peaks(smoothed, smoothed);

				for (int i = 0; i < NBINS; i++) {
					for (int j = 0; j < 192; j++) {
						initial = smoothed[i][j] * multiplier;
						if (previous[i][j] < initial) {
							previous[i][j] = initial;
						}
						else {

						}

					}
				}
				sawtooth_filter(previous, scratch, previous, NBINS);
				// Compute hann

				stft(audio_padded, shifted_hann_window, stft_complex);

				for (int i = 0; i < NBINS; i++) {
					for (int j = 64; j < 128; j++) {
						//apply the mask
						stft_output[i][j] = stft_complex[i][j] * previous[i][j];
					}
				}

				flag = 1; //update the flag because we are good to go
				istft(stft_output);
				return  output;
			}

			if (flag == 1) {
				// Compute product using the residual buffer since on the last round it contained data
				istft(stft_output);

				flag = 2; //update the flag since we processed zeros
				return  output;
			}
		}
		//in the final case, we no longer need to process a residual buffer,
		//since the buffer was zeros on the last run, so no istft call is needed here.
		// Return zero std::array
		return empty;

	};
};






////////////code below here is just for testing purposes
void generate_sine_wave(std::array<float, 8192>& data, float freq, float phase, float amplitude) {
	const float sample_rate = 48000;
	const float delta_phase = 2.0f * M_PI * freq / sample_rate;

	for (int i = 0; i < 8192; i++) {
		data[i] += amplitude * sin(delta_phase * i + phase);
	}
}
#ifndef M_PI_2
#define M_PI_2 (M_PI / 2.0)
#endif

#ifndef M_PI_4
#define M_PI_4 (M_PI / 4.0)
#endif

void generate_complex_wave(std::array<float, 8192>& data) {
	data.fill(0.0f);

	// Generate three sine waves with different frequencies, phases, and amplitudes
	generate_sine_wave(data, 440.0f, 0.0f, 1.0f);
	generate_sine_wave(data, 880.0f, M_PI_2, 0.5f);
	generate_sine_wave(data, 1320.0f, M_PI_4, 0.25f);
}



#include <ctime>

int main() {
	Filter my_filter;
	std::array<float, 8192> demo = { 0 };
	generate_complex_wave(demo);
	std::array<float, 8192> output = { 0 };

	clock_t start_time, end_time;
	start_time = clock(); // get start time

	double sum = 0;
	for (int i = 0; i < 20; i++) {
		output = my_filter.process(demo); // execute the function
		for (int i = 0; i < 8192; i++) {
			sum = sum + abs(output[i]);
		}
	}

	end_time = clock(); // get end time
	float duration = (float)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0; // calculate duration in milliseconds
	std::cout << "Total value " << sum  << std::endl;
	std::cout << "Total execution time: " << duration << " milliseconds" << std::endl;


	return 0;
}
