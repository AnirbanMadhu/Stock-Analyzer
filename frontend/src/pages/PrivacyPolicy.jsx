import React from 'react'
import { Link } from 'react-router-dom'
import { Shield, ArrowLeft } from 'lucide-react'

const PrivacyPolicy = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center mb-4">
          <Shield className="h-12 w-12 text-blue-600" />
        </div>
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Privacy Policy</h1>
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
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Introduction</h2>
          <p className="text-gray-700 leading-relaxed">
            Stock Analyzer ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy 
            explains how we collect, use, disclose, and safeguard your information when you use our stock analysis 
            platform and related services.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Information We Collect</h2>
          
          <h3 className="text-lg font-medium text-gray-900 mb-2">Personal Information</h3>
          <ul className="list-disc list-inside text-gray-700 space-y-1 mb-4">
            <li>Name and email address when you create an account</li>
            <li>Username and password for account authentication</li>
            <li>Profile information you choose to provide</li>
          </ul>

          <h3 className="text-lg font-medium text-gray-900 mb-2">Usage Information</h3>
          <ul className="list-disc list-inside text-gray-700 space-y-1 mb-4">
            <li>Stock symbols you search for and analyze</li>
            <li>Portfolio and watchlist data</li>
            <li>Platform usage patterns and preferences</li>
            <li>Device information and IP address</li>
          </ul>

          <h3 className="text-lg font-medium text-gray-900 mb-2">Financial Data</h3>
          <ul className="list-disc list-inside text-gray-700 space-y-1">
            <li>Investment portfolio information (if you choose to provide it)</li>
            <li>Stock holdings and transaction data (stored locally on your device)</li>
            <li>Investment preferences and risk tolerance</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">How We Use Your Information</h2>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>Provide and maintain our stock analysis services</li>
            <li>Personalize your experience and improve our platform</li>
            <li>Send you important updates about our services</li>
            <li>Provide customer support and respond to your inquiries</li>
            <li>Analyze usage patterns to improve our features</li>
            <li>Ensure platform security and prevent fraud</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Information Sharing</h2>
          <p className="text-gray-700 leading-relaxed mb-4">
            We do not sell, trade, or otherwise transfer your personal information to outside parties except in the following circumstances:
          </p>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>With your explicit consent</li>
            <li>To comply with legal obligations or court orders</li>
            <li>To protect our rights, property, or safety</li>
            <li>With trusted service providers who assist in operating our platform (under strict confidentiality agreements)</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Data Security</h2>
          <p className="text-gray-700 leading-relaxed">
            We implement appropriate technical and organizational security measures to protect your personal information 
            against unauthorized access, alteration, disclosure, or destruction. However, no method of transmission over 
            the internet is 100% secure, and we cannot guarantee absolute security.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Your Rights</h2>
          <p className="text-gray-700 leading-relaxed mb-4">You have the right to:</p>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>Access, update, or delete your personal information</li>
            <li>Opt-out of marketing communications</li>
            <li>Request a copy of your data</li>
            <li>Lodge a complaint with a supervisory authority</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Third-Party Services</h2>
          <p className="text-gray-700 leading-relaxed">
            Our platform uses third-party services for stock data (Yahoo Finance) and analytics. These services have 
            their own privacy policies, and we encourage you to review them. We are not responsible for the privacy 
            practices of these third-party services.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Contact Us</h2>
          <p className="text-gray-700 leading-relaxed">
            If you have any questions about this Privacy Policy, please contact us at:
          </p>
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <p className="text-gray-700">
              <strong>Email:</strong> privacy@stockanalyzer.com<br />
              <strong>Address:</strong> 123 Financial District, New York, NY 10001<br />
              <strong>Phone:</strong> 1-800-STOCK-01
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}

export default PrivacyPolicy
