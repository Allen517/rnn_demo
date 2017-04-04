/**   
 * @package	com.kingwang.cdmrnn.utils
 * @File		AttBatchDerivative.java
 * @Crtdate	Oct 2, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnn.batchderv.impl;

import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;

import com.kingwang.rnn.batchderv.BatchDerivative;
import com.kingwang.rnn.cons.AlgCons;

/**
 *
 * @author King Wang
 * 
 * Oct 2, 2016 8:29:49 PM
 * @version 1.0
 */
public class OutputBatchDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6945194770004264043L;

	public DoubleMatrix dWhy;
	public DoubleMatrix dby;
	
	public void clearBatchDerv() {
		dWhy = null;
		dby = null;
	}

	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		DoubleMatrix _dWhy = acts.get("dWhy");
		DoubleMatrix _dby = acts.get("dby");
		
		if(dWhy==null) {
			dWhy = new DoubleMatrix(_dWhy.rows, _dWhy.columns);
		}
		if(dby==null) {
			dby = new DoubleMatrix(_dby.rows, _dby.columns);
		}
		
		dWhy = dWhy.add(_dWhy).mul(avgFac);
		dby = dby.add(_dby).mul(avgFac);
		
	}

}
