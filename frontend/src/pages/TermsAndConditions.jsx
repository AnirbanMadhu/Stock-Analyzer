import React from 'react'
import { Link } from 'react-router-dom'
import { FileText, ArrowLeft } from 'lucide-react'

const TermsAndConditions = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center mb-4">
          <FileText className="h-12 w-12 text-blue-600" />
        </div>
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Terms & Conditions</h1>
        <p className="text-gray-600">Last updated: August 17, 2025</p>
      </div>

      {/* Back Button */}
      <Link 
        to="/" 
        className="inline-flex items-center text-blue-600 hover:text-blue-800 transition-colors"
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Home
      </Link>

      {/* Content */}
      <div className="bg-white rounded-lg shadow-lg p-8 space-y-6">
        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Acceptance of Terms</h2>
          <p className="text-gray-700 leading-relaxed">
            By accessing and using Stock Analyzer ("the Platform"), you accept and agree to be bound by the terms 
            and provision of this agreement. If you do not agree to abide by the above, please do not use this service.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Description of Service</h2>
          <p className="text-gray-700 leading-relaxed mb-4">
            Stock Analyzer provides stock market analysis tools, portfolio management features, and educational 
            resources. Our services include:
          </p>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>Real-time and historical stock data</li>
            <li>AI-powered stock price predictions</li>
            <li>Portfolio tracking and management tools</li>
            <li>Market news and analysis</li>
            <li>Stock comparison and research tools</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Investment Disclaimer</h2>
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <p className="text-yellow-800 font-medium">
              <strong>IMPORTANT:</strong> Stock Analyzer is for informational and educational purposes only.
            </p>
          </div>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>All information provided is for educational purposes and should not be considered as financial advice</li>
            <li>Past performance does not guarantee future results</li>
            <li>All investments carry risk, and you may lose money</li>
            <li>You should consult with a qualified financial advisor before making investment decisions</li>
            <li>We are not liable for any investment losses you may incur</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">User Account</h2>
          <p className="text-gray-700 leading-relaxed mb-4">
            To access certain features, you may be required to create an account. You agree to:
          </p>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>Provide accurate and complete information</li>
            <li>Maintain the security of your account credentials</li>
            <li>Accept responsibility for all activities under your account</li>
            <li>Notify us immediately of any unauthorized use</li>
            <li>Not share your account with others</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Acceptable Use</h2>
          <p className="text-gray-700 leading-relaxed mb-4">You agree not to:</p>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>Use the platform for any unlawful purpose</li>
            <li>Attempt to gain unauthorized access to our systems</li>
            <li>Interfere with or disrupt our services</li>
            <li>Upload malicious software or content</li>
            <li>Violate any applicable laws or regulations</li>
            <li>Infringe on intellectual property rights</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Data Accuracy</h2>
          <p className="text-gray-700 leading-relaxed">
            While we strive to provide accurate and up-to-date information, we cannot guarantee the accuracy, 
            completeness, or timeliness of any data on our platform. Stock prices and market data may be delayed. 
            Always verify information through official sources before making investment decisions.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Intellectual Property</h2>
          <p className="text-gray-700 leading-relaxed">
            The content, features, and functionality of Stock Analyzer are owned by us and are protected by 
            international copyright, trademark, and other intellectual property laws. You may not reproduce, 
            distribute, or create derivative works without our written permission.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Limitation of Liability</h2>
          <p className="text-gray-700 leading-relaxed">
            To the fullest extent permitted by law, Stock Analyzer shall not be liable for any indirect, incidental, 
            special, consequential, or punitive damages, including but not limited to financial losses, arising from 
            your use of our platform.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Termination</h2>
          <p className="text-gray-700 leading-relaxed">
            We reserve the right to terminate or suspend your account and access to our services at our sole discretion, 
            without prior notice, for any reason, including breach of these terms.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Changes to Terms</h2>
          <p className="text-gray-700 leading-relaxed">
            We may update these terms from time to time. We will notify you of any significant changes by posting 
            the new terms on this page. Your continued use of the platform after changes constitutes acceptance 
            of the new terms.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Contact Information</h2>
          <p className="text-gray-700 leading-relaxed">
            If you have any questions about these Terms & Conditions, please contact us:
          </p>
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <p className="text-gray-700">
              <strong>Email:</strong> legal@stockanalyzer.com<br />
              <strong>Address:</strong> 123 Financial District, New York, NY 10001<br />
              <strong>Phone:</strong> 1-800-STOCK-01
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}

export default TermsAndConditions
