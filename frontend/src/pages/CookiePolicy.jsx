import React from 'react'
import { Cookie, Settings, Shield, Eye, Database } from 'lucide-react'

const CookiePolicy = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-8">
        {/* Header */}
        <div className="flex items-center space-x-3 mb-8">
          <Cookie className="h-8 w-8 text-orange-500" />
          <h1 className="text-3xl font-bold text-gray-900">Cookie Policy</h1>
        </div>

        {/* Introduction */}
        <div className="bg-orange-50 border-l-4 border-orange-500 p-6 mb-8">
          <div className="flex items-center space-x-2 mb-2">
            <Shield className="h-5 w-5 text-orange-500" />
            <h2 className="text-lg font-semibold text-orange-700">About Our Cookie Usage</h2>
          </div>
          <p className="text-orange-700">
            This Cookie Policy explains how Stock Analyzer uses cookies and similar technologies to enhance your experience on our platform.
          </p>
        </div>

        {/* What Are Cookies */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">What Are Cookies?</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              Cookies are small text files stored on your device when you visit our website. They help us:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Remember your preferences and settings</li>
              <li>Keep you logged in during your session</li>
              <li>Analyze how you use our platform</li>
              <li>Provide personalized content and recommendations</li>
              <li>Improve our services and user experience</li>
            </ul>
          </div>
        </section>

        {/* Types of Cookies */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Types of Cookies We Use</h2>
          
          {/* Essential Cookies */}
          <div className="mb-6">
            <div className="flex items-center space-x-2 mb-3">
              <Settings className="h-5 w-5 text-red-500" />
              <h3 className="text-lg font-semibold text-gray-800">Essential Cookies</h3>
              <span className="bg-red-100 text-red-800 text-xs font-medium px-2.5 py-0.5 rounded">Required</span>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-gray-700 mb-3">
                These cookies are necessary for the website to function properly and cannot be disabled.
              </p>
              <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
                <li>Authentication and session management</li>
                <li>Security and fraud prevention</li>
                <li>Core website functionality</li>
                <li>Load balancing and performance</li>
              </ul>
            </div>
          </div>

          {/* Functional Cookies */}
          <div className="mb-6">
            <div className="flex items-center space-x-2 mb-3">
              <Eye className="h-5 w-5 text-blue-500" />
              <h3 className="text-lg font-semibold text-gray-800">Functional Cookies</h3>
              <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded">Optional</span>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-gray-700 mb-3">
                These cookies enhance functionality and personalization but are not essential.
              </p>
              <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
                <li>Remember your preferences (theme, language, currency)</li>
                <li>Save your watchlist and portfolio settings</li>
                <li>Customize dashboard layout</li>
                <li>Remember recently viewed stocks</li>
              </ul>
            </div>
          </div>

          {/* Analytics Cookies */}
          <div className="mb-6">
            <div className="flex items-center space-x-2 mb-3">
              <Database className="h-5 w-5 text-green-500" />
              <h3 className="text-lg font-semibold text-gray-800">Analytics Cookies</h3>
              <span className="bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded">Optional</span>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-gray-700 mb-3">
                These cookies help us understand how visitors interact with our website.
              </p>
              <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
                <li>Track page views and user behavior</li>
                <li>Measure website performance</li>
                <li>Identify popular features and content</li>
                <li>Generate usage statistics and reports</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Third-Party Cookies */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Third-Party Services</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              We use the following third-party services that may set their own cookies:
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded border">
                <h4 className="font-semibold text-gray-800 mb-2">Yahoo Finance</h4>
                <p className="text-sm text-gray-600">
                  Financial data and market information services
                </p>
              </div>
              <div className="bg-white p-4 rounded border">
                <h4 className="font-semibold text-gray-800 mb-2">Google Analytics</h4>
                <p className="text-sm text-gray-600">
                  Website traffic and user behavior analysis
                </p>
              </div>
              <div className="bg-white p-4 rounded border">
                <h4 className="font-semibold text-gray-800 mb-2">CDN Services</h4>
                <p className="text-sm text-gray-600">
                  Content delivery and performance optimization
                </p>
              </div>
              <div className="bg-white p-4 rounded border">
                <h4 className="font-semibold text-gray-800 mb-2">Security Services</h4>
                <p className="text-sm text-gray-600">
                  DDoS protection and security monitoring
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Cookie Management */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Managing Your Cookie Preferences</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Browser Settings</h3>
            <p className="text-gray-700 mb-4">
              You can control cookies through your browser settings:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-6">
              <li><strong>Chrome:</strong> Settings → Privacy and Security → Cookies and other site data</li>
              <li><strong>Firefox:</strong> Preferences → Privacy & Security → Cookies and Site Data</li>
              <li><strong>Safari:</strong> Preferences → Privacy → Manage Website Data</li>
              <li><strong>Edge:</strong> Settings → Cookies and site permissions → Cookies and site data</li>
            </ul>

            <h3 className="text-lg font-semibold text-gray-800 mb-3">Cookie Consent</h3>
            <p className="text-gray-700 mb-4">
              When you first visit our website, you'll see a cookie consent banner where you can:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Accept all cookies for the full experience</li>
              <li>Reject optional cookies (functional and analytics only)</li>
              <li>Customize your preferences for specific cookie types</li>
              <li>Change your preferences at any time through account settings</li>
            </ul>
          </div>
        </section>

        {/* Impact of Disabling Cookies */}
        <div className="bg-yellow-50 border-l-4 border-yellow-500 p-6 mb-8">
          <h2 className="text-lg font-semibold text-yellow-700 mb-3">Impact of Disabling Cookies</h2>
          <p className="text-yellow-700 mb-3">
            Disabling certain cookies may affect your experience:
          </p>
          <ul className="list-disc list-inside space-y-1 text-yellow-700">
            <li>You may need to log in repeatedly</li>
            <li>Your preferences won't be saved</li>
            <li>Some features may not work properly</li>
            <li>Personalized content may not be available</li>
          </ul>
        </div>

        {/* Data Retention */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Cookie Retention Periods</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600 mb-2">Session</div>
                <div className="text-sm text-gray-600">Deleted when browser closes</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600 mb-2">30 Days</div>
                <div className="text-sm text-gray-600">Functional preferences</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600 mb-2">2 Years</div>
                <div className="text-sm text-gray-600">Analytics data</div>
              </div>
            </div>
          </div>
        </section>

        {/* Contact Information */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Questions About Cookies</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              If you have questions about our cookie usage or need help managing your preferences:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Email us at <strong>privacy@stockanalyzer.com</strong></li>
              <li>Visit our Contact page for more support options</li>
              <li>Check your account settings for cookie preferences</li>
            </ul>
          </div>
        </section>

        {/* Footer */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <p className="text-sm text-gray-600">
            <strong>Last Updated:</strong> August 17, 2025
          </p>
          <p className="text-sm text-gray-600 mt-2">
            We may update this Cookie Policy from time to time. Please review it periodically for any changes.
          </p>
        </div>
      </div>
    </div>
  )
}

export default CookiePolicy
