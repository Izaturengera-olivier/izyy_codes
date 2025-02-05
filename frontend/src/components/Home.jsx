import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { ArrowRightIcon } from "@heroicons/react/24/solid";

const Home = () => {
  const [posts, setPosts] = useState([]);

    useEffect(() => {
  axios.get("http://127.0.0.1:8000/posts/")
    .then(response => {
      console.log("API Response:", response.data);
      setPosts(response.data);
    })
    .catch(error => console.error("Error fetching posts:", error));
}, []); // ✅ Dependency array ensures it runs only once


  return (
    <div className="bg-gray-100 min-h-screen font-sans">
      {/* Hero Section */}
      <header className="bg-gray-800 text-white py-20 text-center shadow-md px-6 md:px-12">
        <h1 className="text-4xl font-bold">Welcome to Our Blog</h1>
        <p className="mt-3 text-lg">Explore insightful articles and stay informed</p>
      </header>

      {/* Featured Posts */}
      <section className="container mx-auto px-6 py-12">
        <h2 className="text-3xl font-semibold text-gray-800 mb-6 text-center">Latest Posts</h2>
        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-8">
          {posts.length > 0 ? (
            posts.map((post) => (
              <div key={post.id} className="bg-white rounded-lg shadow-md overflow-hidden">
                {post.featured_image && (
                  <img src={post.featured_image} alt={post.title} className="w-full h-48 object-cover" />
                )}
                <div className="p-4">
                  <h3 className="text-xl font-bold text-gray-900">{post.title}</h3>
                  <p className="text-gray-600 text-sm mt-2">By {post.author}</p>
                  <Link to={`/post/${post.slug}`} className="text-blue-500 flex items-center mt-3 hover:underline font-medium">
                    Read More <ArrowRightIcon className="w-4 h-4 ml-2" />
                  </Link>
                </div>
              </div>
            ))
          ) : (
            <p className="text-center text-gray-600">No posts available</p>
          )}
        </div>
      </section>

      {/* Newsletter Subscription */}
      <section className="bg-gray-700 text-white py-12 text-center px-6">
        <h2 className="text-3xl font-bold">Subscribe to Our Newsletter</h2>
        <p className="mt-3">Get the latest posts delivered straight to your inbox.</p>
        <div className="mt-6 flex flex-col sm:flex-row justify-center">
          <input type="email" placeholder="Enter your email" className="p-3 rounded-t-lg sm:rounded-l-lg sm:rounded-t-none border-none text-gray-800 w-full sm:w-80 focus:ring-2 focus:ring-gray-500" />
          <button className="bg-white text-gray-800 px-5 py-3 rounded-b-lg sm:rounded-r-lg sm:rounded-b-none font-semibold hover:bg-gray-200">Subscribe</button>
        </div>
      </section>
    </div>
  );
};

export default Home;