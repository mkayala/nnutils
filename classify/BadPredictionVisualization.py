#!/usr/bin/env python
# encoding: utf-8
"""
Class to create HTML files of the different reactive Atoms.  

Input data is in format:
"idx" "targ" "pred" "rpred"
137925 -1 16.59429 15965.5
144045 -1 16.09609 15963.5
176865 -1 14.74191 15937.5
125458 -1 14.2665 15921
121997 -1 14.26632 15919.5
59937 -1 14.25898 15914

The idx -> is the reactAtom DB idx.
Want to load in info about this reactAtom, then construct some html visualization of it.

BadPredictionVisualization.py

Created by Matt Kayala on 2010-06-11.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""
import sys
import os
import csv
from optparse import OptionParser;

from CHEM.Common.Smi2Depict import Smi2Depict;
from CHEM.Common.Util import stdOpen, ProgressDots, molBySmiles;
from CHEM.Common.Env import SQL_PLACEHOLDER;
from CHEM.Common import DBUtil;

from CHEM.score.DB.Util import log, orbConnFactory;

from Util import log;

class BadPredictionVisualization:
    """Look at module level documentation"""
    def __init__(self, connFactory=orbConnFactory):
        """Constructor"""
        self.connFactory = connFactory;
        self.conn = None;
        self.depict = Smi2Depict();
        self.depict.depictOpts["width"] = 500; 
        self.depict.depictOpts["height"] = 400; 
        self.depict.depictOpts["lonePairsVisible"] = "true"; 
        self.depict.depictOpts["atomMappingVisible"] = "true"; 
        self.depict.depictOpts["molbg"] = "#ffffff"; 
        self.depict.depictOpts["cgibinDir"] = ".."; 
        self.depict.depictOpts["extraImageSetting"] = "lp,amap";
    
    
    def outFileHeader(self, fileName):
        """Return a header"""
        header = """<?xml version="1.0" encoding="ascii"?>
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
                  "DTD/xhtml1-transitional.dtd">
        <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
        <head>
          <title>Worst predictions : %s</title>
        </head>
        
        <body bgcolor="white" text="black" link="blue" vlink="#204080"
              alink="#204080">
        <h1>Worst predictions : %s </h1>
        <table border=1>
            
        """ % (fileName, fileName)
        return header;
    
    
    def outFileFooter(self):
        footer = """</table>
        </body>
        </html>
        """
        return footer;
    
    
    def main(self, argv):
        """Callable from Command line"""
        if argv is None:
            argv = sys.argv
        
        usageStr = \
            """usage: %prog [options] inputDir outputDir
            """
        
        parser = OptionParser(usage = usageStr);
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 2:
            inputDir, outputDir = args;
            self.conn = self.connFactory.connection();
            
            inFiles = [os.path.join(inputDir, os.path.basename(file)) for file in os.listdir(inputDir)];
            outFiles = [os.path.join(outputDir, '%s.html' % os.path.basename(file)) for file in inFiles];
            
            for iFile, oFile in zip(inFiles, outFiles):
                self.processSingleFile(iFile, oFile);
            
            self.conn.close();
        else:
            parser.print_help();
            sys.exit(2);
    
    
    def processSingleFile(self, inFile, outFile):
        """Given a single file process and create output for it."""
        log.info('Looking at %s ' % os.path.basename(inFile));
        ifs = open(inFile);
        ofs = open(outFile, 'w');
        progress = ProgressDots();
        
        print >>ofs, self.outFileHeader(os.path.basename(inFile));
        
        #First line is simply the header.
        headerData = ifs.next();
        for line in ifs:
            misClassData = self.loadDataSingleLine(line);
            print >>ofs, self.htmlizeSingleData(misClassData);
            progress.Update();
        progress.PrintStatus();
        
        ifs.close();
        ofs.close();
        return;
        
        
    def htmlizeSingleData(self, data):
        """Given some data that has been processed about a particular line, convert to html rep."""
        html = """
        <tr>
            <td>
                %(depictImg)s <br />
                rAtomId = %(idx)d, atomId = %(atomId)d, opReactId = %(opReactantId)d <br />
                rxnConditionsId= %(reactionConditionsId)d, smi = %(atmMapSmi)s <br/>
                target = %(targ)s, prediction = %(pred)s, rank prediction = %(rpred)s <br />
                isPredictedFilled = %(isPredFilled)s, isPredictedUnfilled = %(isPredUnfilled)s 
            </td>
        <tr>
        """ % data;
        return html;
    
    
    def loadDataSingleLine(self, line):
        """Given a sinlge line, return a dictionary of the relevant info to construct the html"""
        chunks = line.strip().split();
        (idx, targ, pred, rpred) = chunks
        idx = int(idx);
        
        dbQry = \
            """
            SELECT ra.atom_id, ra.reaction_conditions_id,
            ra.is_predicted_filled, ra.is_predicted_unfilled,
            a.op_reactant_id, a.atm_map_smi
            FROM reactive_atom ra INNER JOIN atom a
            ON ra.atom_id=a.atom_id
            WHERE ra.reactive_atom_id=%s
            """
        
        #dbQry = """SELECT test_case_id, orb_pair_reactants_id, atm_map_smi, is_predicted_filled, 
        #            is_predicted_unfilled FROM atom WHERE atom_id=%s"""
        res = DBUtil.execute(dbQry, (idx, ), conn=self.conn);
        res = res[0];
        d = {};
        d['idx'] = idx;
        d['targ'] = targ;
        d['pred'] = pred;
        d['rpred'] = rpred;
        d['atomId'] = res[0];
        d['reactionConditionsId'] = res[1]
        d['isPredFilled'] = str(res[2])
        d['isPredUnfilled'] = str(res[3]);
        d['opReactantId'] = res[4];
        d['atmMapSmi'] = res[5]
        d['depictImg'] = self.depict.depictImg(d['atmMapSmi']);
        
        return d;
    
    

if __name__ == '__main__':
    instance = BadPredictionVisualization();
    sys.exit(instance.main(sys.argv));