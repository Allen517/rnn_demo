/**   
 * @package	com.kingwang.cdmrnn.rnn
 * @File		OutputLayer.java
 * @Crtdate	Sep 28, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnn.cell.impl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.rnn.batchderv.BatchDerivative;
import com.kingwang.rnn.batchderv.impl.OutputBatchDerivative;
import com.kingwang.rnn.cell.Cell;
import com.kingwang.rnn.cell.Operator;
import com.kingwang.rnn.comm.utils.FileUtil;
import com.kingwang.rnn.cons.AlgCons;
import com.kingwang.rnn.utils.Activer;
import com.kingwang.rnn.utils.LoadTypes;
import com.kingwang.rnn.utils.MatIniter;
import com.kingwang.rnn.utils.MatIniter.Type;

/**
 *
 * @author King Wang
 * 
 * Sep 28, 2016 5:00:51 PM
 * @version 1.0
 */
public class OutputLayer extends Operator implements Cell, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8868938450690252135L;
	
	public DoubleMatrix Why;
    public DoubleMatrix by;
    
	public DoubleMatrix hdWhy;
    public DoubleMatrix hdby;
    
	public DoubleMatrix hd2Why;
    public DoubleMatrix hd2by;
    
    public OutputLayer(int outSize, int nodeSize, MatIniter initer) {
        if (initer.getType() == Type.Uniform) {
            this.Why = initer.uniform(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
            this.Why = initer.gaussian(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
            this.Why = initer.svd(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        }
        
        this.hdWhy = new DoubleMatrix(outSize, nodeSize);
        this.hdby = new DoubleMatrix(1, nodeSize);
        
        this.hd2Why = new DoubleMatrix(outSize, nodeSize);
        this.hd2by = new DoubleMatrix(1, nodeSize);
        
    }
    
    public void active(int t, Map<String, DoubleMatrix> acts, double... params) {
    	DoubleMatrix haty = yDecode(acts.get("h" + t));
        DoubleMatrix py = Activer.softmax(haty);
        acts.put("py" + t, py);
    }
    
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {
    	DoubleMatrix dWhy = new DoubleMatrix(Why.rows, Why.columns);
    	DoubleMatrix dby = new DoubleMatrix(by.rows, by.columns);
    	
    	for (int t = lastT; t > -1; t--) {
            // delta y
            DoubleMatrix py = acts.get("py" + t);
            DoubleMatrix y = acts.get("y" + t);
            
        	DoubleMatrix deltaY = py.sub(y);
            acts.put("dy" + t, deltaY);

            dWhy = dWhy.add(acts.get("h" + t).transpose().mmul(deltaY));
            dby = dby.add(deltaY);
            
    	}
    	
    	acts.put("dWhy", dWhy);
    	acts.put("dby", dby);
    	
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	OutputBatchDerivative batchDerv = (OutputBatchDerivative) derv;
    	
        hdWhy = hdWhy.add(MatrixFunctions.pow(batchDerv.dWhy, 2.));
        hdby = hdby.add(MatrixFunctions.pow(batchDerv.dby, 2.));
        
        Why = Why.sub(batchDerv.dWhy.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhy).add(eps),-1.).mul(lr)));
        by = by.sub(batchDerv.dby.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdby).add(eps),-1.).mul(lr)));
        
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
    						, double beta1, double beta2, int epochT) {
    	
    	OutputBatchDerivative batchDerv = (OutputBatchDerivative) derv;
    	
		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		hd2Why = hd2Why.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhy, 2.).mul(1 - beta2));
		hd2by = hd2by.mul(beta2).add(MatrixFunctions.pow(batchDerv.dby, 2.).mul(1 - beta2));
		
		hdWhy = hdWhy.mul(beta1).add(batchDerv.dWhy.mul(1 - beta1));
		hdby = hdby.mul(beta1).add(batchDerv.dby.mul(1 - beta1));
		
		Why = Why.sub(
				hdWhy.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Why.mul(biasBeta2)).add(eps), -1))
				);
		by = by.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2by.mul(biasBeta2)).add(eps), -1.)
				.mul(hdby.mul(biasBeta1)).mul(lr)
				);
    }
    
    public DoubleMatrix yDecode(DoubleMatrix ht) {
		return ht.mmul(Why).add(by);
	}
    
	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#writeCellParameter(java.lang.String, boolean)
	 */
	@Override
	public void writeCellParameter(String outFile, boolean isAttached) {
		OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile, isAttached);
    	FileUtil.writeln(osw, "Why");
    	writeMatrix(osw, Why);
    	FileUtil.writeln(osw, "by");
    	writeMatrix(osw, by);
    	FileUtil.writeln(osw, "Whd");
	}

	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#loadCellParameter(java.lang.String)
	 */
	@Override
	public void loadCellParameter(String cellParamFile) {
		LoadTypes type = LoadTypes.Null;
		int row = 0;
		
		try(BufferedReader br = FileUtil.getBufferReader(cellParamFile)) {
			String line = null;
			while((line=br.readLine())!=null) {
				String[] elems = line.split(",");
				if(elems.length<2 && !elems[0].contains(".")) {
    				String typeStr = "Null";
    				String[] typeList = {"Why", "by"};
    				for(String tStr : typeList) {
    					if(elems[0].equalsIgnoreCase(tStr)) {
    						typeStr = tStr;
    						break;
    					}
    				}
    				type = LoadTypes.valueOf(typeStr);
    				row = 0;
    				continue;
    			}
				switch(type) {
					case Why: this.Why = matrixSetter(row, elems, this.Why); break;
					case by: this.by = matrixSetter(row, elems, this.by); break;
				}
				row++;
			}
			
		} catch(IOException e) {
			
		}
	}
}
