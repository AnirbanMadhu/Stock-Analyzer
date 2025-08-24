import React from 'react'
import { Copyright, Shield, AlertCircle } from 'lucide-react'

const Trademark = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-8">
        {/* Header */}
        <div className="flex items-center space-x-3 mb-8">
          <Copyright className="h-8 w-8 text-blue-500" />
          <h1 className="text-3xl font-bold text-gray-900">Trademark Information</h1>
        </div>

        {/* Introduction */}
        <div className="bg-blue-50 border-l-4 border-blue-500 p-6 mb-8">
          <div className="flex items-center space-x-2 mb-2">
            <Shield className="h-5 w-5 text-blue-500" />
            <h2 className="text-lg font-semibold text-blue-700">Intellectual Property Notice</h2>
          </div>
          <p className="text-blue-700">
            This page outlines trademark and intellectual property information for Stock Analyzer and related services.
          </p>
        </div>

        {/* Our Trademarks */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Our Trademarks</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              The following trademarks and service marks are owned by Stock Analyzer or its affiliates:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Stock Analyzer™</strong> - Our primary service mark</li>
              <li><strong>AI Stock Predictor™</strong> - Our artificial intelligence prediction service</li>
              <li><strong>Portfolio Manager Pro™</strong> - Our portfolio management tools</li>
              <li><strong>Market Intelligence Suite™</strong> - Our comprehensive analysis platform</li>
              <li>Stock Analyzer logo and design marks</li>
              <li>Associated graphics, logos, and page headers</li>
            </ul>
          </div>
        </section>

        {/* Third-Party Trademarks */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Third-Party Trademarks</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              We acknowledge the following third-party trademarks used in our service:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Yahoo Finance™</strong> - Trademark of Yahoo Inc., used for data attribution</li>
              <li><strong>NASDAQ®</strong> - Registered trademark of The NASDAQ Stock Market LLC</li>
              <li><strong>NYSE®</strong> - Registered trademark of New York Stock Exchange LLC</li>
              <li><strong>S&P 500®</strong> - Registered trademark of S&P Dow Jones Indices LLC</li>
              <li><strong>Dow Jones®</strong> - Registered trademark of Dow Jones & Company, Inc.</li>
              <li>Various stock ticker symbols and company names are trademarks of their respective owners</li>
            </ul>
            <p className="text-gray-700 mt-4">
              All third-party trademarks, service marks, and logos are the property of their respective owners.
            </p>
          </div>
        </section>

        {/* Usage Guidelines */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Trademark Usage Guidelines</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Permitted Use</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Referencing our services in educational or editorial content</li>
              <li>Using company name in comparative analysis or reviews</li>
              <li>Academic research and non-commercial educational purposes</li>
            </ul>

            <h3 className="text-lg font-semibold text-gray-800 mb-3">Prohibited Use</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Using our trademarks in a way that implies endorsement or affiliation</li>
              <li>Modifying, altering, or creating derivative works of our marks</li>
              <li>Using our trademarks in domain names or business names</li>
              <li>Any use that may cause confusion about the source of goods or services</li>
              <li>Commercial use without express written permission</li>
            </ul>
          </div>
        </section>

        {/* Copyright Information */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Copyright Information</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              © 2025 Stock Analyzer. All rights reserved.
            </p>
            <p className="text-gray-700 mb-4">
              The content, design, graphics, compilation, magnetic translation, digital conversion, 
              and other matters related to this website are protected under applicable copyrights, 
              trademarks, and other proprietary rights.
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Website design and user interface elements</li>
              <li>Proprietary algorithms and analysis methodologies</li>
              <li>Educational content and tutorials</li>
              <li>Software code and technical implementations</li>
              <li>Marketing materials and documentation</li>
            </ul>
          </div>
        </section>

        {/* Reporting Violations */}
        <div className="bg-yellow-50 border-l-4 border-yellow-500 p-6 mb-8">
          <div className="flex items-center space-x-2 mb-2">
            <AlertCircle className="h-5 w-5 text-yellow-500" />
            <h2 className="text-lg font-semibold text-yellow-700">Reporting Trademark Violations</h2>
          </div>
          <p className="text-yellow-700 mb-3">
            If you believe your trademark rights have been violated or if you notice unauthorized use of our trademarks:
          </p>
          <ul className="list-disc list-inside space-y-1 text-yellow-700">
            <li>Contact our legal team at <strong>legal@stockanalyzer.com</strong></li>
            <li>Provide detailed information about the alleged violation</li>
            <li>Include evidence of trademark ownership or authorization</li>
          </ul>
        </div>

        {/* Legal Disclaimer */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Legal Disclaimer</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              This trademark information is provided for general guidance only and does not constitute legal advice. 
              Trademark laws vary by jurisdiction and are subject to change.
            </p>
            <p className="text-gray-700">
              For specific legal questions regarding trademark use or intellectual property matters, 
              please consult with a qualified intellectual property attorney.
            </p>
          </div>
        </section>

        {/* Footer */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <p className="text-sm text-gray-600">
            <strong>Last Updated:</strong> August 17, 2025
          </p>
          <p className="text-sm text-gray-600 mt-2">
            For questions about trademark usage or licensing opportunities, please contact us at 
            <strong> legal@stockanalyzer.com</strong>
          </p>
        </div>
      </div>
    </div>
  )
}

export default Trademark
