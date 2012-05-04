#!/usr/bin/env python
# encoding: utf-8
"""
Class to turn information about redundant reactove_atoms into visualizations

RedundDataVisualization.py

Created by Matt Kayala on 2010-11-01.
Copyright (c) 2010 Institute for Genomics and Bioinformatics. All rights reserved.
"""

import sys
import os
from optparse import OptionParser;

from CHEM.Common.Smi2Depict import Smi2Depict;
from CHEM.Common.Util import stdOpen, ProgressDots, molBySmiles;
from CHEM.Common.Env import SQL_PLACEHOLDER;
from CHEM.Common import DBUtil;

from CHEM.score.DB.Util import log, orbConnFactory;

from Util import log;

class RedundDataVisualization:
    """Turn redundant reactve atom data into simple visualizations"""
    def __init__(self, connFactory=orbConnFactory):
        """Constructor"""
        self.connFactory = connFactory;
        self.conn = None;
        self.depict = Smi2Depict();
        self.depict.depictOpts["width"] = 300; 
        self.depict.depictOpts["height"] = 200; 
        self.depict.depictOpts["lonePairsVisible"] = "true"; 
        self.depict.depictOpts["atomMappingVisible"] = "true"; 
        self.depict.depictOpts["molbg"] = "#ffffff"; 
        self.depict.depictOpts["cgibinDir"] = ".."; 
        self.depict.depictOpts["extraImageSetting"] = "lp,amap";
    
    
    def outFileHeader(self):
        """Return a header"""
        header = """<?xml version="1.0" encoding="ascii"?>
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
                  "DTD/xhtml1-transitional.dtd">
        <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
        <head>
          <title>Redundant predictions</title>
        </head>
        
        <body bgcolor="white" text="black" link="blue" vlink="#204080"
              alink="#204080">
        <h1>redundant predictions</h1>
        <table border=1>
            
        """ 
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
            """usage: %prog [options] inputFile outputFile
            
            inputfile is of format:
            posAtomId redundNegAtomId
            
            output is an html file with depictions of these atoms next
            to each other
            
            """
        
        parser = OptionParser(usage = usageStr);
        (options, args) = parser.parse_args(argv[1:])
        
        if len(args) == 2:
            inputFile, outputFile = args;
            self.conn = self.connFactory.connection();
            ifs = open(inputFile)
            ofs = open(outputFile, 'w')
            
            print >>ofs, self.outFileHeader();
            
            for line in ifs:
                posData, negData = self.loadDataSingleLine(line);
                print >>ofs, self.htmlizeSingleData(posData, negData);
            
            
            print >>ofs, self.outFileFooter();
            ifs.close();
            ofs.close();
            self.conn.close();
        else:
            parser.print_help();
            sys.exit(2);
    
    
    def htmlizeSingleData(self, posData, negData):
        """Given some data that has been processed about a particular line, convert to html rep."""
        cellHtml = """
            <td>
                %(depictImg)s <br />
                reactiveAtomId= %(reactive_atom_id)d, atomId = %(atom_id)d, opReactId = %(op_reactant_id)d <br />
                smi = %(atm_map_smi)s, reactionConditionsId= %(reaction_conditons_id)d <br/>
                isPredictedFilled = %(is_predicted_filled)s, isPredictedUnfilled = %(is_predicted_unfilled)s 
            </td>
        """
        
        html = """
        <tr>
            %s
            %s
        <tr>
        """ % (cellHtml % posData, cellHtml % negData);
        return html;
    
    
    def loadDataSingleLine(self, line):
        """Given a sinlge line, return a dictionary of the relevant info to construct the html"""
        chunks = line.strip().split();
        (posIdx, negIdx) = chunks
        posIdx = int(posIdx)
        negIdx = int(negIdx)
        
        colnames = \
            [
                'atom_id',
                'reaction_conditons_id',
                'is_predicted_filled',
                'is_predicted_unfilled',
                'op_reactant_id',
                'atm_map_smi'
            ]
        qry = \
            """SELECT ra.atom_id, ra.reaction_conditions_id,
                ra.is_predicted_filled, ra.is_predicted_unfilled,
                a.op_reactant_id, a.atm_map_smi
                FROM reactive_atom ra INNER JOIN atom a
                ON ra.atom_id=a.atom_id
                WHERE ra.reactive_atom_id=%s
            """
        log.info('Lookign up posIdx: %d' % posIdx)
        posRes = DBUtil.execute(qry, (posIdx, ), conn=self.conn, connFactory=self.connFactory)[0]
        log.info('Lookign up negIdx: %d' % negIdx)
        negRes = DBUtil.execute(qry, (negIdx, ), conn=self.conn, connFactory=self.connFactory)[0]
        
        pDict = {};
        nDict = {};
        for iCol in range(len(posRes)):
            pDict[colnames[iCol]] = posRes[iCol]
            nDict[colnames[iCol]] = negRes[iCol]
        
        pDict['reactive_atom_id'] = posIdx
        nDict['reactive_atom_id'] = negIdx;
        pDict['depictImg'] = self.depict(pDict['atm_map_smi'])
        nDict['depictImg'] = self.depict(nDict['atm_map_smi'])
        
        return (pDict, nDict);
    
    



if __name__ == '__main__':
    instance = RedundDataVisualization();
    sys.exit(instance.main(sys.argv));