'use client'; 
export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-200 py-20 px-6">
      <div className="max-w-5xl mx-auto space-y-16">

        {/* Page Title */}
        <div className="text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-white">
            About Our Helmet Detection System
          </h1>
          <p className="mt-4 text-lg text-gray-400">
            Smart, Fast & Accurate AI-powered Safety Monitoring
          </p>
        </div>

        {/* Section 1 */}
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <h2 className="text-2xl font-semibold text-white mb-4">
              Our Mission
            </h2>
            <p className="text-gray-400 leading-relaxed">
              Our goal is to make workplaces safer using cutting-edge Artificial
              Intelligence. Helmet Detection helps companies automatically identify
              whether workers are following safety rules by wearing helmets.
              This reduces workplace injuries and improves compliance monitoring.
            </p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 text-center">
            <span className="text-6xl">🛡️</span>
            <p className="text-gray-400 mt-3">AI-Enhanced Safety Technology</p>
          </div>
        </div>

        {/* Section 2 */}
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 text-center md:order-last">
            <span className="text-6xl">⚙️</span>
            <p className="text-gray-400 mt-3">Smart Detection Models</p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold text-white mb-4">
              How It Works
            </h2>
            <p className="text-gray-400 leading-relaxed">
              Our system uses advanced deep learning models to analyze images or
              live camera feeds. The AI detects whether a helmet is present and
              provides real-time feedback. Accuracy improves over time through
              continuous machine learning updates.
            </p>
          </div>
        </div>

        {/* Section 3 */}
        <div className="text-center space-y-4">
          <h2 className="text-3xl font-semibold">Why Choose Us?</h2>

          <ul className="text-gray-300 space-y-3 md:text-lg max-w-xl mx-auto">
            <li>✔️ Fast and accurate helmet detection</li>
            <li>✔️ Modern, user-friendly interface</li>
            <li>✔️ Real-time monitoring for industries</li>
            <li>✔️ Easy integration with existing systems</li>
            <li>✔️ Reduces safety risks significantly</li>
          </ul>
        </div>

      </div>
    </div>
  );
}







